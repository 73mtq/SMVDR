# 在正确的运行目录下跑通 test.py 的推理流程
#
# 用法：
#   pwsh -File repro\scripts\run_test.ps1              # 逐字运行 test.py
#   pwsh -File repro\scripts\run_test.ps1 -UseDriver   # 改用驱动脚本
#
# 两种模式的差别：
#   逐字运行执行的是仓库原生 test.py，最接近论文的命令。测试集只有 12 只眼，
#   主进程加载数据足够快。
#   驱动脚本 repro\scripts\smoke_test.py 复用 test.py 里的 test_model2，
#   只把 DataLoader 的 num_workers 与批量交出来，并额外核对 280 个键是否
#   全部命中模型（堵住 create_models.py:30-33 的静默失败），最后打印显存峰值。
#
# 历史记录：test.py:198 原本写死 num_workers=24。Windows 用 spawn 方式启动子进程，
# 每个子进程都会重新执行 test.py 顶层并导入 sklearn/scipy。24 份子进程在这台
# 15.7 GB 内存的机器上实测崩溃，worker 内报
#   ImportError: DLL load failed while importing _flapack: 页面文件太小，无法完成操作。
# 父进程退出码 3221225725（0xC00000FD，STATUS_STACK_OVERFLOW）。已把该处改为
# num_workers=0，见 repro\README.md 第 9.3 节。

param(
    [switch]$UseDriver,
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [switch]$NoMask,
    [string]$Name = 'smoke'
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

& $Py -c "import sys; print('python', sys.version.split()[0])"

New-Item -ItemType Directory -Force -Path (Join-Path $RunDir 'weights') | Out-Null

$srcWeight = Join-Path $RepoRoot 'weights\best_smvdr.pth'
$dstWeight = Join-Path $RunDir 'weights\best_smvdr.pth'
if (-not (Test-Path $srcWeight)) {
    throw "找不到权重 $srcWeight，请先按 README 第 5 节放置官方权重"
}
Copy-Item $srcWeight $dstWeight -Force
Write-Host "权重       : $dstWeight ($((Get-Item $dstWeight).Length) 字节)"

foreach ($d in 'test_process', 'test_mask') {
    $p = Join-Path $DataRoot $d
    if (-not (Test-Path $p)) {
        Write-Warning "缺少 $p"
    }
}

Push-Location $RunDir
try {
    if ($UseDriver) {
        Write-Host ''
        Write-Host '=== 驱动脚本模式 ==='
        $driverArgs = @(
            (Join-Path $RepoRoot 'repro\scripts\smoke_test.py'),
            '--batch-size', $BatchSize,
            '--num-workers', $NumWorkers,
            '--name', $Name
        )
        if ($NoMask) { $driverArgs += '--no-mask' }
        & $Py @driverArgs
    }
    else {
        Write-Host ''
        Write-Host '=== 逐字运行 test.py ==='
        & $Py (Join-Path $RepoRoot 'test.py')
    }
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '=== 输出文件 ==='
Get-ChildItem $RunDir -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

if ($code -ne 0) {
    throw "运行结束，退出码 $code。按顺序排查：数据目录 $DataRoot 下的 4 个目录是否齐备、权重是否在位、显存与内存是否被别的程序占满。"
}
Write-Host '运行成功。'
