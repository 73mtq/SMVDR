import argparse
import math
import os
import sys

import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transform
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import models.create_models as create
import train as train_module
from dataset import MultiviewImgDataset_mask
from models.FLoss import FocalLoss
from models.mamba import _ratio_loss
from models.moe import entropy_regularization_loss
from train import seed_everything, val_model

DATA_PATH = '../../EYData_BaseEye_newdata/'
TRAIN_PATH = '../../EYData_BaseEye_newdata/train_process'
MASK_PATH = '../../EYData_BaseEye_newdata/train_mask'
TEST_PATH = '../../EYData_BaseEye_newdata/test_process'
MASK_PATH2 = '../../EYData_BaseEye_newdata/test_mask'

IMAGE_SIZE = 224
LR = 0.00001
DEPTH = 12
HEAD = 9
SEED = 42
GRAD_KEYS = ('trans_cls_head.final_fc.weight', 'conv1.weight')


def main():
    parser = argparse.ArgumentParser(description='用真实样例图跑通 train.py 的训练步骤')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--batches', type=int, default=4, help='本次跑多少个批次')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--name', default='smoke_train')
    parser.add_argument('--no-mask', action='store_true',
                        help='使用 dataset.py 自带的 no_mask 开关，把掩膜像素全部覆盖为 255')
    args = parser.parse_args()

    print('工作目录 :', os.getcwd())
    print('数据目录 :', os.path.abspath(DATA_PATH))

    seed_everything(SEED)
    print('随机种子 :', SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('设备     :', device)

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.0),
        transform.RandomVerticalFlip(p=0.0),
        transform.RandomResizedCrop(IMAGE_SIZE),
        transform.ToTensor(),
    ])
    transform_valid = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
    ])

    train_csv_path = os.path.join(DATA_PATH, 'train_rgb_label_newname.csv')
    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_dataset = MultiviewImgDataset_mask(TRAIN_PATH, MASK_PATH, train_df,
                                             transform=transform_train, Single=False, no_mask=args.no_mask)
    valid_dataset = MultiviewImgDataset_mask(TEST_PATH, MASK_PATH2, test_df,
                                             transform=transform_valid, Single=False, no_mask=args.no_mask)
    print('训练样本 :', len(train_dataset), ' 验证样本 :', len(valid_dataset))

    # train.val_model 内部引用的是模块级全局 valid_dataset 与 NUM_VIEW（train.py:173），
    # 这两个名字定义在 train.py 的 if __name__ == '__main__': 块里，
    # 直接 import train 时不会执行那一段，所以必须在这里补上，否则 val_model 会抛 NameError。
    train_module.valid_dataset = valid_dataset
    train_module.NUM_VIEW = 1
    print('已注入 train.valid_dataset 与 train.NUM_VIEW（train.py:173 的 val_model 依赖它们）')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    model = create.my_SMVDR(
        in_chans=4,
        pretrained=False,
        num_classes=5,
        pre_Path='unused',
        depth=DEPTH,
        num_heads=HEAD,
        drop_rate=0.1,
        drop_path_rate=0.1,
        cuda=device,
    )
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    criterion = FocalLoss(gamma=2)
    named = dict(model.named_parameters())

    print('参数量   :', sum(p.numel() for p in model.parameters()))
    print('批量     :', args.batch_size, ' 批次上限 :', args.batches)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for index, (img, label) in enumerate(train_loader):
        if index >= args.batches:
            break
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        if tuple(img.shape[1:3]) != (4, 4):
            raise SystemExit('输入形状异常：{}，期望第二维是 4 个视角、第三维是 4 个通道'.format(tuple(img.shape)))

        batch, views, channels, height, width = img.size()
        imgs_mix = img.view(-1, channels, height, width)
        evidences, expert_probabilities, selector = model(imgs_mix)
        loss = criterion(evidences, label) + 0.1 * _ratio_loss(True, selector, ratio=0.7) \
            + 0.1 * entropy_regularization_loss(expert_probabilities)

        if not math.isfinite(loss.item()):
            raise SystemExit('第 {} 个批次的损失不是有限值：{}'.format(index, loss.item()))

        optimizer.zero_grad()
        loss.backward()

        for key in GRAD_KEYS:
            grad = named[key].grad
            if grad is None:
                raise SystemExit('{} 的梯度是 None，反向传播没有生效'.format(key))
            if not torch.isfinite(grad).all():
                raise SystemExit('{} 的梯度不是有限值'.format(key))

        optimizer.step()
        scheduler.step(index / max(1, args.batches))
        optimizer.step()

        print('批次 {}  loss {:.4f}  lr {:.8f}  显存 {:.0f} MiB'.format(
            index, loss.item(), optimizer.param_groups[-1]['lr'],
            torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0))

    lr_before = optimizer.param_groups[-1]['lr']
    scheduler.step(2.0)
    lr_after = optimizer.param_groups[-1]['lr']
    if lr_before == lr_after:
        raise SystemExit('调度器步进后学习率没有变化：{} -> {}'.format(lr_before, lr_after))
    print('调度器   : lr {:.8f} -> {:.8f}'.format(lr_before, lr_after))

    val_acc, val_loss = val_model(model, valid_loader, criterion, device)
    val_acc = float(val_acc)
    val_loss = float(val_loss)
    if not math.isfinite(val_acc) or not math.isfinite(val_loss):
        raise SystemExit('验证返回值不是有限值：acc {} loss {}'.format(val_acc, val_loss))
    print('验证     : acc {:.4f} loss {:.4f}'.format(val_acc, val_loss))

    if torch.cuda.is_available():
        print('显存峰值 : {:.1f} MiB'.format(torch.cuda.max_memory_allocated() / 1024 / 1024))

    print()
    print('完成。训练链路（前向、三项损失、反向、优化器、调度器、验证）全部执行成功。')


if __name__ == '__main__':
    main()
