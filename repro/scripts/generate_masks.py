import hashlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# huggingface_hub 在 import 时读取这些环境变量，必须在 import 之前设置。
# HF_ENDPOINT：本机 huggingface.co 的 DNS 解析失败，改用 hf-mirror.com。
# HF_HUB_DISABLE_XET：hf-xet 会绕过 HF_ENDPOINT 直连 huggingface.co，必须关掉。
# HF_HOME：把权重缓存放进 repro\_work 下，该目录已被 .gitignore 覆盖。
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HOME'] = os.path.join(REPO_ROOT, 'repro', '_work', 'hf-cache')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, REPO_ROOT)

from process_image import crop_image_from_gray
from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.functional import autofit_fundus_resolution
from fundus_lesions_toolkit.constants import LESIONS
from fundus_lesions_toolkit.models.hf_hub import HuggingFaceModel

WORK = os.path.join(REPO_ROOT, 'repro', '_work')
SAMPLE_DIR = os.path.join(WORK, 'samples')
PROCESSED_DIR = os.path.join(WORK, 'processed')
MODEL_DIR = os.path.join(WORK, 'hf')

SPLITS = ('train', 'test')

MODEL_REPO = 'ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d'
MODEL_FILENAME = 'model.safetensors'
# 与 HuggingFace 上该文件的 LFS oid 一致，见 repro\README.md 第 6 节
MODEL_SHA256 = 'a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2'

ARCH = 'unet'
ENCODER = 'seresnext50_32x4d'
IMAGE_RESOLUTION = 1024
MASK_SIZE = 1280
CROP_TOL = 7
RAW_SHAPE = (2056, 2124)

# toolkit 的类别顺序是 LESIONS = ["BG", "CTW", "EX", "HE", "MA"]，取值 0..4。
# SMVDR 的掩膜约定是 0=背景、1=EX（硬性渗出）、2=MA（微血管瘤）、3=SE（软性渗出，即 CTW 棉绒斑）、4=HE（出血）。
# 下表下标是 toolkit 的类别编号，值是 SMVDR 的掩膜取值。
LESION_TO_SMVDR = np.array([0, 3, 1, 4, 2], dtype=np.uint8)

N_CLASSES = 5


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights():
    # 侦察阶段已经下载过一份，存在且校验通过就直接复用，避免再传 138 MB。
    local = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if os.path.exists(local) and sha256_of(local) == MODEL_SHA256:
        print('  权重文件   : {}（复用已下载）'.format(local))
        return local
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    actual = sha256_of(path)
    if actual != MODEL_SHA256:
        raise SystemExit('{} 的 sha256 是 {}，期望 {}。下载到的不是官方权重。'.format(
            path, actual, MODEL_SHA256))
    print('  权重文件   : {}'.format(path))
    return path


def load_model(path, device):
    # 权重键名带 "model." 前缀，说明它是外层 HuggingFaceModel 的 state_dict，
    # 所以按 toolkit 的方式构造外层对象，再把权重严格载入它内部那个 torchseg 模型。
    # 不用 toolkit 的 from_pretrained：那里 strict 默认 False，键名对不上会静默跳过。
    wrapper = HuggingFaceModel(arch=ARCH, encoder=ENCODER, in_channels=3, classes=N_CLASSES)
    weights = load_file(path)
    state_dict = wrapper.state_dict()
    missing = sorted(k for k in weights if k not in state_dict)
    if missing:
        raise SystemExit('权重里有 {} 个键不在模型里：{}'.format(len(missing), missing[:8]))
    absent = sorted(k for k in state_dict if k not in weights)
    if absent:
        raise SystemExit('模型里有 {} 个键在权重里找不到：{}'.format(len(absent), absent[:8]))
    mismatch = [k for k in weights if tuple(state_dict[k].shape) != tuple(weights[k].shape)]
    if mismatch:
        raise SystemExit('有 {} 个键形状不一致：{}'.format(len(mismatch), mismatch[:8]))
    wrapper.load_state_dict(weights, strict=True)
    print('  键名核对   : {} / {} 全部命中，形状全部一致'.format(len(weights), len(state_dict)))

    model = wrapper.model.to(device)
    model.eval()
    return model


def segment_padded(model, padded, device):
    # 与 toolkit 的 segment(autofit_resolution=False, reverse_autofit=False) 等价：
    # 归一化到 [0,1]、按 IMAGENET 均值方差标准化、前向、softmax 到概率。
    image = (padded / 255.0).astype(np.float32)
    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)
    mean, std = get_normalization()
    tensor = Ftv.normalize(tensor, mean=mean, std=std)
    with torch.inference_mode():
        pred = F.softmax(model(tensor), 1)
    if tuple(pred.shape) != (1, N_CLASSES, IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        raise SystemExit('分割输出的形状是 {}。'.format(tuple(pred.shape)))
    return pred.squeeze(0)


def gray_crop_indices(rgb):
    # 复刻 process_image.crop_image_from_gray 的裁剪判定。
    # 该函数不返回索引、只返回裁剪后的图，所以这里重算一遍索引，
    # 再由调用方与 crop_image_from_gray 的实际输出逐像素比对。
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > CROP_TOL
    rows = mask.any(1)
    cols = mask.any(0)
    if rgb[:, :, 0][np.ix_(rows, cols)].shape[0] == 0:
        # 原函数在图像过暗时会退回整幅，这里保持同样的分支
        rows = np.ones(rgb.shape[0], dtype=bool)
        cols = np.ones(rgb.shape[1], dtype=bool)
    return rows, cols


def reverse_to_original(pred_padded, transforms, raw_shape):
    # toolkit 的 reverse_autofit_tensor 用 round(尺寸 * 1/f) 反算裁剪区，
    # 四舍五入可能让结果比原图多一列或少一列，掩膜就会与 train_process 错位。
    # 这里把目标尺寸直接取成原图尺寸减去上下左右边缘，得到精确的原始坐标图。
    crop = transforms['crop']
    pad_t, pad_b, pad_l, pad_r = transforms['pad']
    target_h = raw_shape[0] - pad_t - pad_b
    target_w = raw_shape[1] - pad_l - pad_r
    inner = pred_padded[:, crop[0]:crop[1], crop[2]:crop[3]]
    inner = F.interpolate(inner.unsqueeze(0), size=(target_h, target_w),
                          mode='bilinear', align_corners=True).squeeze(0)
    full = F.pad(inner, pad=(pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
    if tuple(full.shape) != (N_CLASSES, raw_shape[0], raw_shape[1]):
        raise SystemExit('反算后的形状是 {}，期望 {}。'.format(
            tuple(full.shape), (N_CLASSES, raw_shape[0], raw_shape[1])))
    return full


def build_mask(model, rgb, device, stats):
    padded, transforms = autofit_fundus_resolution(rgb, IMAGE_RESOLUTION)
    if padded.shape[:2] != (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        raise SystemExit('自动裁剪画布尺寸是 {}，期望 ({}, {})。'.format(
            padded.shape[:2], IMAGE_RESOLUTION, IMAGE_RESOLUTION))

    pred = segment_padded(model, padded, device)
    pred = reverse_to_original(pred, transforms, rgb.shape[:2])
    classes = torch.argmax(pred, dim=0).to(torch.uint8).cpu().numpy()

    cropped = crop_image_from_gray(rgb)
    rows, cols = gray_crop_indices(rgb)
    repro = rgb[np.ix_(rows, cols)]
    if repro.shape != cropped.shape or not np.array_equal(repro, cropped):
        raise SystemExit('复刻的裁剪索引与 process_image.crop_image_from_gray 的输出不一致：'
                         '复刻 {}，原函数 {}。'.format(repro.shape, cropped.shape))

    cropped_classes = classes[np.ix_(rows, cols)]
    if cropped_classes.shape != cropped.shape[:2]:
        raise SystemExit('裁剪后的类别图是 {}，预处理图是 {}。'.format(
            cropped_classes.shape, cropped.shape))

    # 先换成 SMVDR 约定再缩放，最近邻插值不会混合类别值
    mapped = LESION_TO_SMVDR[cropped_classes]
    mask = cv2.resize(mapped, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)

    if mask.shape != (MASK_SIZE, MASK_SIZE):
        raise SystemExit('掩膜形状是 {}，期望 ({}, {})。'.format(mask.shape, MASK_SIZE, MASK_SIZE))
    if mask.dtype != np.uint8:
        raise SystemExit('掩膜 dtype 是 {}。'.format(mask.dtype))
    unknown = np.setdiff1d(np.unique(mask), np.arange(N_CLASSES, dtype=np.uint8))
    if unknown.size:
        raise SystemExit('掩膜出现了约定之外的取值：{}。'.format(unknown.tolist()))

    for value in range(N_CLASSES):
        stats[value] += int((mask == value).sum())
    return mask


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('工作目录   : {}'.format(REPO_ROOT))
    print('缓存目录   : {}'.format(os.environ['HF_HOME']))
    print('HF_ENDPOINT: {}'.format(os.environ['HF_ENDPOINT']))
    print('设备       : {}'.format(device))
    print('类别顺序   : {} -> SMVDR {}'.format(LESIONS, LESION_TO_SMVDR.tolist()))

    if not os.path.isdir(SAMPLE_DIR):
        raise SystemExit('找不到 {}，请先执行 repro\\scripts\\prepare_smoke_data.py'.format(SAMPLE_DIR))

    print()
    print('=== 分割模型 ===')
    model_path = fetch_weights()
    model = load_model(model_path, device)

    stats = [0] * N_CLASSES
    total = 0
    for split in SPLITS:
        src = os.path.join(SAMPLE_DIR, split)
        dst = os.path.join(PROCESSED_DIR, '{}_mask'.format(split))
        os.makedirs(dst, exist_ok=True)
        names = sorted(n for n in os.listdir(src) if n.endswith('.jpg'))
        if not names:
            raise SystemExit('{} 里没有 jpg 文件：{}'.format(split, src))
        total += len(names)
        print()
        print('=== {} ==='.format(split))
        print('  待生成 {} 张 -> {}'.format(len(names), dst))

        for index, name in enumerate(names, 1):
            rgb = cv2.cvtColor(cv2.imread(os.path.join(src, name)), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != RAW_SHAPE:
                raise SystemExit('{} 的尺寸是 {}，期望 {}。'.format(name, rgb.shape[:2], RAW_SHAPE))
            mask = build_mask(model, rgb, device, stats)
            out_path = os.path.join(dst, '{}_mask.png'.format(name[:-4]))
            if not cv2.imwrite(out_path, mask):
                raise SystemExit('写入失败：{}'.format(out_path))
            if index % 8 == 0 or index == len(names):
                print('  完成 {}/{}'.format(index, len(names)))

        written = sorted(n for n in os.listdir(dst) if n.endswith('_mask.png'))
        if len(written) != len(names):
            raise SystemExit('{} 输出 {} 张掩膜，输入 {} 张图。'.format(split, len(written), len(names)))

    print()
    print('=== 类别直方图（全部 {} 张） ==='.format(total))
    total_pixels = sum(stats)
    for value, count in enumerate(stats):
        print('  {}  {:>13d}  {:7.4f}%'.format(value, count, 100.0 * count / total_pixels))

    if stats[1] + stats[3] + stats[4] == 0:
        raise SystemExit('EX/SE/HE 三类像素总数为 0，掩膜里没有任何病灶标记。')
    for value in (1, 2, 3, 4):
        if stats[value] == 0:
            print('  警告：类别 {} 没有像素。'.format(value))

    print()
    print('完成。掩膜在 {}'.format(PROCESSED_DIR))


if __name__ == '__main__':
    main()
