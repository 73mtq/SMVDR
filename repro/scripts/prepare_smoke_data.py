import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = os.path.join(REPO_ROOT, 'repro', '_work')
LABEL_DIR = os.path.join(WORK, 'mfiddr_labels')
SAMPLE_DIR = os.path.join(WORK, 'samples')
PROCESSED_DIR = os.path.join(WORK, 'processed')

SPLITS = ('train', 'test')
REPO_CSV = '{0}_rgb_label_newname.csv'
LABEL_URL = 'https://raw.githubusercontent.com/mfiddr/MFIDDR/main/sample/{}_fourpic_label.csv'
LIST_URL = 'https://api.github.com/repos/mfiddr/MFIDDR/contents/sample/{}-examples'
IMAGE_URL = 'https://raw.githubusercontent.com/mfiddr/MFIDDR/main/sample/{}-examples/{}'


def fetch(url, retries=3):
    last = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'smvdr-repro'})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            print('    第 {} 次请求失败：{}'.format(attempt, exc))
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError('请求失败，已尝试 {} 次：{}'.format(retries, url)) from last


def download_labels():
    os.makedirs(LABEL_DIR, exist_ok=True)
    for split in SPLITS:
        path = os.path.join(LABEL_DIR, '{}_fourpic_label.csv'.format(split))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print('  已存在，跳过：{}'.format(os.path.basename(path)))
            continue
        data = fetch(LABEL_URL.format(split))
        with open(path, 'wb') as handle:
            handle.write(data)
        print('  写入 {:>9d} 字节  {}'.format(len(data), os.path.basename(path)))


def download_samples():
    for split in SPLITS:
        out_dir = os.path.join(SAMPLE_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        listing = json.loads(fetch(LIST_URL.format(split)))
        names = sorted(item['name'] for item in listing if item['name'].endswith('.jpg'))
        before = set(os.listdir(out_dir))
        fetched = 0
        for name in names:
            if name in before:
                continue
            data = fetch(IMAGE_URL.format(split, name))
            with open(os.path.join(out_dir, name), 'wb') as handle:
                handle.write(data)
            fetched += 1
        count = len([n for n in os.listdir(out_dir) if n.endswith('.jpg')])
        print('  {:<6s} 官方 {} 张，新下载 {} 张，目录现有 {} 张'.format(split, len(names), fetched, count))


def flatten_official(rows):
    flat = []
    for row in rows:
        for key in ('id1', 'id2', 'id3', 'id4'):
            flat.append((row[key], row['level']))
    return flat


def build_rename_map():
    mapping = {}
    for split in SPLITS:
        with open(os.path.join(LABEL_DIR, '{}_fourpic_label.csv'.format(split)), encoding='utf-8') as handle:
            official = flatten_official(list(csv.DictReader(handle)))
        with open(os.path.join(REPO_ROOT, REPO_CSV.format(split)), encoding='utf-8') as handle:
            repo = [(row['id'], row['level']) for row in csv.DictReader(handle)]
        if len(official) != len(repo):
            raise RuntimeError('{} 行数不一致：官方 {} 行，仓库 {} 行'.format(split, len(official), len(repo)))
        changed = 0
        for index, ((off, off_level), (rep, rep_level)) in enumerate(zip(official, repo)):
            off_parts = off.split('_')
            rep_parts = rep.split('_')
            if len(off_parts) != 4 or len(rep_parts) != 4:
                raise RuntimeError('{} 第 {} 行文件名不是 4 段：{} / {}'.format(split, index, off, rep))
            if off_parts[0] != rep_parts[0] or off_parts[2] != rep_parts[2] or off_parts[3] != rep_parts[3]:
                raise RuntimeError('{} 第 {} 行三段不一致：官方 {}，仓库 {}'.format(split, index, off, rep))
            if off_level != rep_level:
                raise RuntimeError('{} 第 {} 行等级不一致：官方 {}，仓库 {}'.format(split, index, off_level, rep_level))
            mapping[off] = rep
            if off != rep:
                changed += 1
        print('  {:<6s} 核对通过：{} 行，其中 {} 行需要改名'.format(split, len(official), changed))
    return mapping


def rename_samples(mapping):
    renamed = 0
    for split in SPLITS:
        out_dir = os.path.join(SAMPLE_DIR, split)
        for name in sorted(n for n in os.listdir(out_dir) if n.endswith('.jpg')):
            stem = name[:-4]
            if stem not in mapping:
                raise RuntimeError('{} 的文件 {} 不在官方标签文件里'.format(split, name))
            target = mapping[stem] + '.jpg'
            if target == name:
                continue
            target_path = os.path.join(out_dir, target)
            if os.path.exists(target_path):
                raise RuntimeError('目标文件已存在，拒绝覆盖：{}'.format(target_path))
            os.replace(os.path.join(out_dir, name), target_path)
            renamed += 1
        count = len([n for n in os.listdir(out_dir) if n.endswith('.jpg')])
        print('  {:<6s} 目录现有 {} 张，累计改名 {} 个'.format(split, count, renamed))
    return renamed


def write_subset_csv():
    for split in SPLITS:
        src = os.path.join(SAMPLE_DIR, split)
        want = {n[:-4] for n in os.listdir(src) if n.endswith('.jpg')}
        with open(os.path.join(REPO_ROOT, REPO_CSV.format(split)), encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames)
            rows = list(reader)
        keep = [row for row in rows if row['id'] in want]
        found = {row['id'] for row in keep}
        if found != want:
            missing = sorted(want - found)
            raise RuntimeError('{} 有 {} 个 id 在仓库 CSV 里找不到：{}'.format(split, len(missing), missing[:5]))
        if len(keep) % 4 != 0:
            raise RuntimeError('{} 行数 {} 不是 4 的倍数'.format(split, len(keep)))
        for start in range(0, len(keep), 4):
            block = keep[start:start + 4]
            if len({row['level'] for row in block}) != 1:
                raise RuntimeError('{} 第 {} 组 4 行的 level 不一致'.format(split, start))
            if len({'_'.join(row['id'].split('_')[:1] + row['id'].split('_')[2:3]) for row in block}) != 1:
                raise RuntimeError('{} 第 {} 组 4 行不属于同一只眼'.format(split, start))
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, REPO_CSV.format(split))
        with open(out_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(keep)
        print('  {:<6s} 写入 {} 行 -> {}'.format(split, len(keep), out_path))


def main():
    print('[1/4] 下载官方标签文件（覆盖全部 8613 只眼的完整标签）')
    download_labels()
    print('[2/4] 下载官方样例图（每边 48 张，共 96 张）')
    download_samples()
    print('[3/4] 按官方标签建立改名映射并改名（官方用随机编号，仓库用出生日期）')
    mapping = build_rename_map()
    renamed = rename_samples(mapping)
    print('[4/4] 生成 12 只样例眼的子集标签 CSV')
    write_subset_csv()
    print()
    print('完成。改名 {} 个文件。子集 CSV 在 {}'.format(renamed, PROCESSED_DIR))
    print('下一步：执行 repro\\scripts\\preprocess_samples.py 生成 1280x1280 的预处理图。')


if __name__ == '__main__':
    main()
