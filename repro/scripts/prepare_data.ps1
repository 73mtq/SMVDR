# 把 repro\_work 里准备好的样例数据组装成 train.py / test.py 需要的目录结构
#
# 用法：
#   pwsh -File repro\scripts\prepare_data.ps1
#
# 依次执行三件事：
#   1) repro\scripts\prepare_smoke_data.py  下载官方标签与 96 张真实样例图，改名，生成子集 CSV
#   2) repro\scripts\preprocess_samples.py  用仓库自带的 process_image.py 预处理成 1280x1280
#   3) 把结果复制到 D:\EYData_BaseEye_newdata（train_process / test_process / train_mask / test_mask 四个目录与两份 CSV）
#
# 掩膜由 repro\scripts\run_masks.ps1 生成，必须先跑它，本脚本只负责组装与校验。
#
# 目录位置由 train.py:184-188 与 test.py:176-178 决定：
#   代码里写的是 "../../EYData_BaseEye_newdata/"，按当前工作目录解析。
#   从 D:\github-repository\SMVDR 往上是 D:\github-repository，再往上是 D:\，
#   所以实际目录是 D:\EYData_BaseEye_newdata。

param(
    [switch]$SkipDownload,
    [string]$Python = 'D:\ProgramData\anaconda3\envs\smvdr\python.exe'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = $Python
$DataRoot = 'D:\EYData_BaseEye_newdata'
$Work = Join-Path $RepoRoot 'repro\_work'

Write-Host '=== 环境 ==='
Write-Host "仓库根目录 : $RepoRoot"
Write-Host "解释器     : $Py"
Write-Host "目标数据目录 : $DataRoot"

if (-not (Test-Path $Py)) {
    throw "找不到解释器 $Py，请先按 README 第 2、3 节装好依赖"
}

& $Py -c "import cv2, numpy, pandas; print('cv2', cv2.__version__, ' numpy', numpy.__version__, ' pandas', pandas.__version__)"

if (-not $SkipDownload) {
    Write-Host ''
    Write-Host '=== 第 1 步：下载官方数据并改名 ==='
    & $Py (Join-Path $RepoRoot 'repro\scripts\prepare_smoke_data.py')
    if ($LASTEXITCODE -ne 0) { throw "prepare_smoke_data.py 退出码 $LASTEXITCODE" }
}
else {
    Write-Host ''
    Write-Host '=== 第 1 步：已跳过（-SkipDownload） ==='
}

Write-Host ''
Write-Host '=== 第 2 步：预处理成 1280x1280 ==='
& $Py (Join-Path $RepoRoot 'repro\scripts\preprocess_samples.py')
if ($LASTEXITCODE -ne 0) { throw "preprocess_samples.py 退出码 $LASTEXITCODE" }

Write-Host ''
Write-Host '=== 第 3 步：组装到目标数据目录 ==='
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null

$pairs = @(
    @{ Src = (Join-Path $Work 'processed\train_process'); Dst = (Join-Path $DataRoot 'train_process') },
    @{ Src = (Join-Path $Work 'processed\test_process'); Dst = (Join-Path $DataRoot 'test_process') },
    @{ Src = (Join-Path $Work 'processed\train_mask'); Dst = (Join-Path $DataRoot 'train_mask') },
    @{ Src = (Join-Path $Work 'processed\test_mask'); Dst = (Join-Path $DataRoot 'test_mask') },
    @{ Src = (Join-Path $Work 'processed\train_rgb_label_newname.csv'); Dst = (Join-Path $DataRoot 'train_rgb_label_newname.csv') },
    @{ Src = (Join-Path $Work 'processed\test_rgb_label_newname.csv'); Dst = (Join-Path $DataRoot 'test_rgb_label_newname.csv') }
)

foreach ($p in $pairs) {
    if (-not (Test-Path $p.Src)) { throw "缺少 $($p.Src)" }
    if (Test-Path $p.Dst) { Remove-Item $p.Dst -Recurse -Force }
    Copy-Item $p.Src $p.Dst -Recurse -Force
    $leaf = Split-Path -Leaf $p.Dst
    if (Test-Path $p.Dst -PathType Container) {
        $n = (Get-ChildItem $p.Dst -File).Count
        Write-Host ("  {0,-32} {1} 个文件" -f $leaf, $n)
    }
    else {
        Write-Host ("  {0,-32} {1} 字节" -f $leaf, (Get-Item $p.Dst).Length)
    }
}

Write-Host ''
Write-Host '=== 结果 ==='
Get-ChildItem $DataRoot | Select-Object Mode, Name | Format-Table -AutoSize

Write-Host ''
Write-Host '=== 校验 ==='
foreach ($d in 'train_process', 'test_process', 'train_mask', 'test_mask') {
    $p = Join-Path $DataRoot $d
    if (-not (Test-Path $p)) { throw "缺少目录 $p" }
}
foreach ($f in 'train_rgb_label_newname.csv', 'test_rgb_label_newname.csv') {
    $p = Join-Path $DataRoot $f
    if (-not (Test-Path $p)) { throw "缺少文件 $p" }
}

foreach ($split in 'train', 'test') {
    $imgDir = Join-Path $DataRoot "${split}_process"
    $maskDir = Join-Path $DataRoot "${split}_mask"
    $images = @(Get-ChildItem $imgDir -Filter '*.jpg' -File)
    $masks = @(Get-ChildItem $maskDir -Filter '*_mask.png' -File)
    if ($images.Count -ne $masks.Count) {
        throw "$split 的图 $($images.Count) 张、掩膜 $($masks.Count) 张，数量不一致"
    }
    foreach ($img in $images) {
        $expected = Join-Path $maskDir ($img.BaseName + '_mask.png')
        if (-not (Test-Path $expected)) { throw "缺掩膜 $expected" }
    }
    Write-Host ("  {0,-8} 图 {1} 张，掩膜 {2} 张，逐张对应" -f $split, $images.Count, $masks.Count)
}

Write-Host ''
Write-Host '组装完成，四个目录与两份 CSV 齐备。'
