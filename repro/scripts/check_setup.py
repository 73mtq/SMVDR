import os
import platform
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

EXPECTED_PARAM_NUMEL = 108990864
EXPECTED_STATE_DICT_NUMEL = 108994066
EXPECTED_CKPT_ENTRIES = 280
WEIGHT_NAMES = ('best_smvdr.pth',)


def section(title):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


def find_weight():
    candidates = [os.path.join(REPO_ROOT, 'weights', name) for name in WEIGHT_NAMES]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    section('1. 解释器')
    print('python      :', sys.executable)
    print('version     :', sys.version.replace('\n', ' '))
    print('machine     :', platform.machine())
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('需要 Python 3.11，当前是 {}.{}。请用 D:\\ProgramData\\anaconda3\\envs\\smvdr\\python.exe'.format(
            *sys.version_info[:2]))
    print('检查通过：版本是 3.11')

    section('2. 第三方包')
    import cv2
    import einops
    import matplotlib
    import mmcv
    import numpy
    import pandas
    import sklearn
    import timm
    import torch
    import torchvision
    import tqdm

    packages = [
        ('torch', torch.__version__),
        ('torchvision', torchvision.__version__),
        ('numpy', numpy.__version__),
        ('pandas', pandas.__version__),
        ('opencv-python', cv2.__version__),
        ('scikit-learn', sklearn.__version__),
        ('tqdm', tqdm.__version__),
        ('matplotlib', matplotlib.__version__),
        ('einops', einops.__version__),
        ('timm', timm.__version__),
        ('mmcv', mmcv.__version__),
    ]
    for name, version in packages:
        print('{:<16s} {}'.format(name, version))

    if torch.__version__.split('+')[0] != '2.4.1' or 'cu121' not in torch.__version__:
        raise SystemExit('torch 应为 2.4.1+cu121，实际 {}'.format(torch.__version__))
    if pandas.__version__ != '1.5.3':
        raise SystemExit('pandas 必须为 1.5.3（test.py 用了 pandas 2.0 已移除的 DataFrame.append），实际 {}'.format(
            pandas.__version__))
    if timm.__version__ != '1.0.30':
        raise SystemExit('timm 应为 1.0.30，实际 {}'.format(timm.__version__))
    if mmcv.__version__ != '2.2.0':
        raise SystemExit('mmcv 应为 2.2.0（由 mmcv-lite 提供），实际 {}'.format(mmcv.__version__))
    print('检查通过：torch / pandas / timm / mmcv 版本都符合源码要求')

    section('3. CUDA')
    available = torch.cuda.is_available()
    print('torch.version.cuda :', torch.version.cuda)
    print('cuda.is_available():', available)
    if not available:
        raise SystemExit('CUDA 不可用。可能装成了 CPU 版本，请重新执行 torch 的安装命令')
    device_name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print('device             :', device_name)
    print('total memory       : {:.0f} MiB'.format(total / 1024 / 1024))
    print('capability         :', torch.cuda.get_device_capability(0))

    section('4. 仓库模块导入')
    import models.create_models as create
    print('models.create_models 导入成功')

    section('5. 构建模型并数参数量')
    model = create.my_SMVDR(
        in_chans=4,
        pretrained=False,
        num_classes=5,
        pre_Path='unused',
        depth=12,
        num_heads=9,
        drop_rate=0.0,
        drop_path_rate=0.0,
        cuda='cpu',
    )
    # parameters() 只数可训练参数，state_dict() 还包含 BatchNorm 的 running_mean /
    # running_var / num_batches_tracked。这份权重是 state_dict 格式，所以两个数都要核。
    params = sum(p.numel() for p in model.parameters())
    state_numel = sum(v.numel() for v in model.state_dict().values())
    entries = len(model.state_dict())
    print('parameters() numel :', params)
    print('state_dict() numel :', state_numel)
    print('state_dict() entries:', entries)
    if params != EXPECTED_PARAM_NUMEL:
        raise SystemExit('parameters() 应为 {}，实际 {}'.format(EXPECTED_PARAM_NUMEL, params))
    if state_numel != EXPECTED_STATE_DICT_NUMEL:
        raise SystemExit('state_dict() 应为 {}，实际 {}'.format(EXPECTED_STATE_DICT_NUMEL, state_numel))
    if entries != EXPECTED_CKPT_ENTRIES:
        raise SystemExit('state_dict() 条目应为 {}，实际 {}'.format(EXPECTED_CKPT_ENTRIES, entries))
    print('检查通过：可训练参数量与 state_dict 条目数都与官方权重一致')

    section('6. 官方权重')
    weight_path = find_weight()
    if weight_path is None:
        raise SystemExit('找不到 {}，请放到 {}\\weights\\ 下'.format(WEIGHT_NAMES[0], REPO_ROOT))
    size = os.path.getsize(weight_path)
    print('path               :', weight_path)
    print('bytes              :', size)
    checkpoint = torch.load(weight_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise SystemExit('权重不是 state_dict（是 {}），create_models.py 会走 model = checkpoint 分支'.format(
            type(checkpoint).__name__))
    print('type               :', type(checkpoint).__name__)
    print('entries            :', len(checkpoint))
    print("has 'model' key    :", 'model' in checkpoint)
    if len(checkpoint) != EXPECTED_CKPT_ENTRIES:
        raise SystemExit('权重条目应为 {}，实际 {}'.format(EXPECTED_CKPT_ENTRIES, len(checkpoint)))

    state_dict = model.state_dict()
    matched = [k for k in checkpoint if k in state_dict]
    print('matched keys       : {} / {}'.format(len(matched), len(checkpoint)))
    for key, value in checkpoint.items():
        if key in state_dict and tuple(state_dict[key].shape) != tuple(value.shape):
            raise SystemExit('{} 形状不一致：权重 {}，模型 {}'.format(
                key, tuple(value.shape), tuple(state_dict[key].shape)))
    if len(matched) != len(checkpoint):
        missing = [k for k in checkpoint if k not in state_dict]
        raise SystemExit('有 {} 个键没命中模型：{}'.format(len(missing), missing[:8]))
    print('检查通过：{} 个键全部命中，形状全部一致'.format(len(matched)))

    total_weight_params = sum(v.numel() for v in checkpoint.values())
    print('weight numel       :', total_weight_params)
    if total_weight_params != EXPECTED_STATE_DICT_NUMEL:
        raise SystemExit('权重张量元素总数应为 {}，实际 {}'.format(
            EXPECTED_STATE_DICT_NUMEL, total_weight_params))

    section('全部检查通过')


if __name__ == '__main__':
    main()
