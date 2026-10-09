import os
import sys

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from process_image import preprocess_image

WORK = os.path.join(REPO_ROOT, 'repro', '_work')
SAMPLE_DIR = os.path.join(WORK, 'samples')
PROCESSED_DIR = os.path.join(WORK, 'processed')

SPLITS = ('train', 'test')


def main():
    if not os.path.isdir(SAMPLE_DIR):
        raise SystemExit('找不到 {}，请先执行 repro\\scripts\\prepare_smoke_data.py'.format(SAMPLE_DIR))

    for split in SPLITS:
        src = os.path.join(SAMPLE_DIR, split)
        dst = os.path.join(PROCESSED_DIR, '{}_process'.format(split))
        os.makedirs(dst, exist_ok=True)
        names = sorted(n for n in os.listdir(src) if n.endswith('.jpg'))
        if not names:
            raise SystemExit('{} 里没有 jpg 文件：{}'.format(split, src))
        print('  {:<6s} 待处理 {} 张'.format(split, len(names)))

        for name in names:
            enhanced_rgb, _ = preprocess_image(os.path.join(src, name), sigmaX=15)
            if enhanced_rgb.shape != (1280, 1280, 3):
                raise SystemExit('{} 的输出尺寸是 {}，期望 (1280, 1280, 3)'.format(name, enhanced_rgb.shape))
            ok = cv2.imwrite(os.path.join(dst, name), cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise SystemExit('写入失败：{}'.format(os.path.join(dst, name)))

        written = sorted(n for n in os.listdir(dst) if n.endswith('.jpg'))
        if written != names:
            raise SystemExit('{} 输出文件与输入不一致'.format(split))
        probe = cv2.imread(os.path.join(dst, names[0]))
        print('  {:<6s} 完成 {} 张，抽查 {} 尺寸 {}'.format(split, len(written), names[0], probe.shape))

    print()
    print('完成。预处理图在 {}'.format(PROCESSED_DIR))


if __name__ == '__main__':
    main()
