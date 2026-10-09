# 用官方公开的真实样例图跑通 train.py 的训练步骤
#
# 用法：
#   pwsh -File repro\scripts\run_train.ps1                       # 驱动脚本，批量 2、跑 4 个批次
#   pwsh -File repro\scripts\run_train.ps1 -BatchSize 1 -Batches 2
#   pwsh -File repro\scripts\run_train.ps1 -Verbatim              # 逐字运行 train.py（100 轮）
#   pwsh -File repro\scripts\run_train.ps1 -NoMask
#
# 默认走驱动脚本，原因是 train.py 的三处硬编码按本机显存与内存调整后，
# 批量与轮数仍然取的是固定值，驱动脚本才能在跑通链路的前提下把范围压到几个批次。
#   train.py:202  BATCH_SIZE=2（原为 4），每步 8 张图穿过主干网络
#   train.py:199  N_EPOCHS=100
#   train.py:244-246  num_workers=0（原为 8）
#
# -Verbatim 调用的是仓库原生 train.py，跑满 100 轮。它不接受 -BatchSize、
# -Batches、-NumWorkers，这三个值在 train.py 里是硬编码的。产物 best_smvdr.pth
# 写在运行目录 repro\_work\out 下，不会覆盖 weights\best_smvdr.pth 里的官方权重。
#
# 历史记录：train.py 原本写死 num_workers=8。test.py 在同样机制下的 24 个 worker
# 已实测崩溃（worker 内 ImportError: DLL load failed while importing _flapack:
# 页面文件太小，无法完成操作。，父进程退出码 3221225725），train.py 的三处一并改为 0。
#
# 驱动脚本 repro\scripts\smoke_train.py 复用 train.py 里的 seed_everything 与
# val_model、models\FLoss.py 的 FocalLoss、models\mamba.py 的 _ratio_loss、
# models\moe.py 的 entropy_regularization_loss，只把批量、轮数、工作进程数
# 交出来，并加上断言。

param(
    [int]$BatchSize = 2,
    [int]$Batches = 4,
    [int]$NumWorkers = 0,
    [switch]$Verbatim,
    [switch]$NoMask,
    [string]$Name = 'smoke_train'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = 'D:\ProgramData\anaconda3\envs\smvdr\python.exe'
$DataRoot = 'D:\EYData_BaseEye_newdata'
$RunDir = Join-Path $RepoRoot 'repro\_work\out'

Write-Host '=== 环境 ==='
Write-Host "仓库根目录 : $RepoRoot"
Write-Host "解释器     : $Py"
Write-Host "数据目录   : $DataRoot"
Write-Host "运行目录   : $RunDir"

if (-not (Test-Path $Py)) {
    throw "找不到解释器 $Py，请先按 README 建好 conda 环境 smvdr"
}
if (-not (Test-Path $DataRoot)) {
    throw "找不到数据目录 $DataRoot，请先按 README 第 8 节组装数据"
}

foreach ($d in 'train_process', 'train_mask', 'test_process', 'test_mask') {
    $p = Join-Path $DataRoot $d
    if (-not (Test-Path $p)) {
        throw "缺少 $p。train.py 需要 train_process、train_mask、test_process、test_mask 四个目录"
    }
}

& $Py -c "import sys; print('python', sys.version.split()[0])"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

Push-Location $RunDir
try {
    if ($Verbatim) {
        Write-Host ''
        Write-Host '=== 逐字运行 train.py ==='
        Write-Host '-Verbatim 下 -BatchSize、-Batches、-NumWorkers、-NoMask 都不生效，'
        Write-Host '这四个值在 train.py 里是硬编码的：BATCH_SIZE=2、N_EPOCHS=100、num_workers=0、no_mask=False。'
        Write-Host '要向驱动脚本交参数，去掉 -Verbatim。'
        & $Py (Join-Path $RepoRoot 'train.py')
    }
    else {
        Write-Host ''
        Write-Host '=== 训练路径驱动 ==='
        $driverArgs = @(
            (Join-Path $RepoRoot 'repro\scripts\smoke_train.py'),
            '--batch-size', $BatchSize,
            '--batches', $Batches,
            '--num-workers', $NumWorkers,
            '--name', $Name
        )
        if ($NoMask) { $driverArgs += '--no-mask' }
        & $Py @driverArgs
    }
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($code -ne 0) {
    throw "运行结束，退出码 $code。若报显存不足，把 -BatchSize 调小（例如 1）后重试；逐字运行时报显存不足，先关掉占用显存的程序再试。"
}
Write-Host '运行成功。'
