import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import models.create_models as create
from dataset import MultiviewImgDataset_mask
from test import test_model2

DATA_PATH = '../../EYData_BaseEye_newdata/'
TEST_PATH = '../../EYData_BaseEye_newdata/test_process'
MASK_PATH2 = '../../EYData_BaseEye_newdata/test_mask'
MODELPATH = './weights/best_smvdr.pth'


def main():
    parser = argparse.ArgumentParser(description='用官方权重跑通 test.py 的推理流程')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-mask', action='store_true',
                        help='使用 dataset.py 自带的 no_mask 开关，把掩膜像素全部覆盖为 255')
    parser.add_argument('--name', default='smoke')
    args = parser.parse_args()

    print('工作目录 :', os.getcwd())
    print('数据目录 :', os.path.abspath(DATA_PATH))
    print('权重     :', os.path.abspath(MODELPATH))

    if not os.path.exists(MODELPATH):
        raise SystemExit('找不到权重文件：{}'.format(os.path.abspath(MODELPATH)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('设备     :', device)

    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
    ])

    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')
    test_df = pd.read_csv(test_csv_path)
    print('测试标签 :', test_csv_path, '{} 行'.format(len(test_df)))

    test_dataset = MultiviewImgDataset_mask(TEST_PATH, MASK_PATH2, test_df, transform=transform_test,
                                            Single=False, no_mask=args.no_mask)
    print('测试样本 :', len(test_dataset), '（每样本 4 个视角）')

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

    model = create.my_SMVDR(
        in_chans=4,
        pretrained=True,
        num_classes=5,
        pre_Path=MODELPATH,
        depth=12,
        num_heads=9,
        drop_rate=0.0,
        drop_path_rate=0.0,
        cuda=device,
    )

    checkpoint = torch.load(MODELPATH, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise SystemExit('权重不是 state_dict（是 {}），无法做键名覆盖检查'.format(type(checkpoint).__name__))
    state_dict = model.state_dict()
    matched = [k for k in checkpoint if k in state_dict]
    print('键名覆盖 : {} / {}'.format(len(matched), len(checkpoint)))
    if len(matched) != len(checkpoint):
        missing = [k for k in checkpoint if k not in state_dict]
        raise SystemExit('有 {} 个键没命中，create_models.py 会静默忽略这些权重：{}'.format(
            len(missing), missing[:8]))
    mismatch = [k for k in checkpoint if tuple(state_dict[k].shape) != tuple(checkpoint[k].shape)]
    if mismatch:
        raise SystemExit('有 {} 个键形状不一致：{}'.format(len(mismatch), mismatch[:8]))
    print('形状核对 : {} 个键全部一致'.format(len(matched)))

    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    test_model2(args.name, model, test_loader, len(test_dataset),
                nn.CrossEntropyLoss(reduction='sum'), device)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print('显存峰值 : {:.1f} MiB'.format(peak))

    print()
    print('完成。本目录下的 {}_test_result.txt 与 {}_roc_data.csv 是本次输出。'.format(args.name, args.name))


if __name__ == '__main__':
    main()
