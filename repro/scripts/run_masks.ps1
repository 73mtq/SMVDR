# 为 repro\_work\samples 里的样例图生成病灶掩膜
#
# 用法：
#   pwsh -File repro\scripts\run_masks.ps1
#
# 掩膜是用替代分割模型生成的，不是论文用的 HACDR-Net：
#   模型   ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d
#   架构   unet + seresnext50_32x4d，torchseg 实现
#   类别   ["BG","CTW","EX","HE","MA"] -> SMVDR [0,3,1,4,2]
#   原因   公开渠道没有 HACDR-Net 权重，见 repro\README.md 第 6 节
#
# 输出：repro\_work\processed\train_mask\ 与 test_mask\，各 48 张 1280x1280 单通道 PNG。
# 生成之后用 prepare_data.ps1 把它们组装进 D:\EYData_BaseEye_newdata。

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = 'D:\ProgramData\anaconda3\envs\smvdr\python.exe'
$Work = Join-Path $RepoRoot 'repro\_work'

Write-Host '=== 环境 ==='
Write-Host "仓库根目录 : $RepoRoot"
Write-Host "解释器     : $Py"
Write-Host "样本目录   : $Work\samples"

if (-not (Test-Path $Py)) {
    throw "找不到解释器 $Py，请先按 README 第 2、3 节装好依赖"
}
if (-not (Test-Path (Join-Path $Work 'samples'))) {
    throw "找不到 $Work\samples，请先执行 repro\scripts\prepare_smoke_data.py"
}

& $Py -c "import sys; print('python', sys.version.split()[0])"

& $Py -c "import fundus_lesions_toolkit, os; print('fundus_lesions_toolkit 来自', os.path.dirname(fundus_lesions_toolkit.__file__))"
if ($LASTEXITCODE -ne 0) {
    throw 'fundus_lesions_toolkit 不可用。它必须从 GitHub 装（PyPI 版的权重放在 Google Drive，本机拿不到）。'
}

Write-Host ''
Write-Host '=== 生成掩膜 ==='
& $Py (Join-Path $RepoRoot 'repro\scripts\generate_masks.py')
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "generate_masks.py 退出码 $code"
}

Write-Host ''
Write-Host '=== 输出文件 ==='
foreach ($split in 'train', 'test') {
    $dir = Join-Path $Work "processed\${split}_mask"
    $files = Get-ChildItem $dir -Filter '*_mask.png' -File
    Write-Host ("  {0,-12} {1} 个掩膜" -f "${split}_mask", $files.Count)
}

$probe = Join-Path $Work 'processed\test_mask\3404_19820909_left_05_mask.png'
& $Py -c "import cv2, numpy as np; m = cv2.imread(r'$probe', 0); print('  抽查', m.shape, m.dtype, '取值', sorted(np.unique(m).tolist()))"

Write-Host ''
Write-Host '掩膜已完成。下一步：pwsh -File repro\scripts\prepare_data.ps1 -SkipDownload'
