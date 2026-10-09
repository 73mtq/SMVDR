# SMVDR 复现任务 · 交接状态文件

> 给接手的新 AI 会话：本文件是上一个会话（2026-10-08 22:12 ~ 10-09 11:02）的完整状态快照。
> 接手后**先读完本文件再行动**，已验证过的事实不要重查，已完成的步骤不要重做。
> 每完成一个阶段，把结果更新到本文件末尾的「进度日志」，然后可以放心丢弃旧上下文。

---

## 0. 任务一句话

在 Windows 本机复现 AAAI 2025 论文 SMVDR（*Like an Ophthalmologist: Dynamic Selection Driven Multi-View Learning for Diabetic Retinopathy Grading*）的推理与训练链路。

- 仓库：`D:\github-repository\SMVDR`（官方源码，git 跟踪；`test.py` 与 `train.py` 按本机配置改了 5 处数值，见 §8）
- 复现辅助物（脚本/依赖清单/手册）：仓库 `repro\` 目录，共 16 个文件
- 完整操作手册：`repro\README.md`（13 章，每条命令可直接复制执行）

## 1. 硬事实库（全部已实测核验，勿重查）

| 项 | 事实 |
| --- | --- |
| 机器 | RTX 4060 Laptop（8188 MiB，计算能力 8.9）/ i7-12650H / 15.7 GB 内存 / Windows + PowerShell 7 |
| 内存与页面文件 | 可见物理内存 16,502,656 KB（约 15.74 GB）；`C:\pagefile.sys` 8,704 MB；提交上限 25,415,552 KB（约 24.24 GB）。这组数值决定了 `num_workers` 的取值，见 §8 |
| conda 环境 | `smvdr` = Python 3.11.9，位于 `D:\ProgramData\anaconda3\envs\smvdr`，依赖**已装完**（torch 2.4.1+cu121、torchvision 0.19.1+cu121、numpy 1.26.4、pandas 1.5.3、timm 1.0.30、mmcv 2.2.0 等，共 107 个包，清单见 `repro\env\requirements-lock.txt`） |
| 官方权重 | `D:\桌面\毕业设计\best_smvdr.pth`，436,079,327 字节；zip CRC 全过；data.pkl sha256 `0347526f...fa4d5` 与官方 Google Drive 版一致；280 键 / 张量元素合计 108,994,066（可训练参数 108,990,864 + 6 个 BatchNorm buffer 3,202）；键名 280/280 命中模型。已复制到 `D:\github-repository\SMVDR\weights\best_smvdr.pth` |
| 该权重的身份 | SMVDR-M 变体，论文 84.01% acc，与仓库 `final_test_result.txt` 吻合 |
| 数据集 | MFIDDR（DDR 官方样例），样例 96 张已下载、改名、预处理 1280×1280 全部通过；运行目录 `D:\EYData_BaseEye_newdata` 已组装完毕（train/test 的 process 与 mask 各 48 张 + 两份 CSV，逐张对应校验通过），仓库内 `repro\EYData_BaseEye_newdata` 目录联接指向它 |
| 网络限制 | download.pytorch.org DNS 解析失败 → 用阿里云镜像；huggingface.co 在 Python 侧 DNS 解析失败 → 走 hf-mirror.com，已实测可用（96 张掩膜全部生成） |
| conda init | 本机从未执行过；`conda activate` 不可用，**一律用绝对路径调用解释器** |

正确的解释器调用方式（所有命令共用）：

```powershell
$PY = 'D:\ProgramData\anaconda3\envs\smvdr\python.exe'
```

## 2. pip 报错已定位（2026-10-09 已查清，勿重查）

用户在 PowerShell 直接敲 `python.exe -m pip install ...` 命中 PATH 上的系统 Python 3.12.2（`C:\Users\29602\AppData\Local\Programs\Python\Python312\python.exe`），而 wheel 文件名里的 `cp311` 只适用 CPython 3.11，pip 判定标签不匹配 → `is not a supported wheel on this platform`。

已做 A/B 验证：同一个 wheel，`python.exe`（3.12.2）报错；`D:\ProgramData\anaconda3\envs\smvdr\python.exe`（3.11.9）正常解析为 `Would install`。

**修法**：先 `$PY = 'D:\ProgramData\anaconda3\envs\smvdr\python.exe'` 再 `& $PY -m pip install ...`。

## 3. 掩膜问题：调查结论与当前采用的方案

### 3.1 核心事实

- 权重文件里**没有掩膜**（280 个条目全是参数张量，`conv1.weight` 形状 `(64,4,7,7)` → 输入 4 通道 = RGB + 掩膜通道；掩膜是输入，不是参数）
- MFIDDR 数据集本身不含掩膜；论文用 HACDR-Net 生成（论文称 OIR agent）
- **HACDR-Net 官方权重全网不存在**：遍历 3,471 个路径、7 次提交历史、4 个 fork、releases，0 个权重文件；issue #1/#3/#4 索要均无回复
- 没有掩膜时 test.py 在 `dataset.py:284` 抛 `AttributeError: 'NoneType' object has no attribute 'reshape'`（预期行为）
- `no_mask=True` 可跑通链路 = 论文 w/o prompts 消融（81.64% vs 84.01%），掩膜的价值约 2.4 个百分点

### 3.2 已确定的掩膜来源：替代分割模型

2026-10-09 下午确认采用 `ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d`（HuggingFace，MIT，138,428,676 字节）。

原有的三条路线里，**路线 C 已经实现**，A/B 保留为可选项：

| 路线 | 成本 | 保真度 | 状态 |
| --- | --- | --- | --- |
| A. 发邮件向作者要 HACDR-Net checkpoint（`xqh51199597@outlook.com`，Qihao Xu） | 低（等待） | 最高 | 未发 |
| B. 自训 HACDR-Net（OIA-DDR 数据 + 40000 iters） | 高：另建 Python 3.10 + torch 1.12.0+cu116 + mmcv-full 1.6.2 环境，依赖已记录在 `repro\env\requirements-hacdr.txt` | 高 | 未启动 |
| C. 替代分割模型（HuggingFace `ClementP/...`） | 已查清 | 中：DDR 上 EX 0.6117 / HE 0.5448 / MA 0.2341 / CTW 0.5085，掩膜质量低于 HACDR-Net | **已实现，脚本就绪** |

路线 C 的关键事实（全部实测核验）：

- 模型 444 个张量 / 34,594,820 个参数；`model.decoder.blocks.*` 的形状与 torchseg `Unet` 默认参数（`decoder_channels=(256,128,64,32,16)`、`encoder_depth=5`）完全吻合
- 类别顺序 `LESIONS = ["BG","CTW","EX","HE","MA"]`，映射到 SMVDR 约定是 `[0,3,1,4,2]`（0→0，1 CTW/SE→3，2 EX→1，3 HE→4，4 MA→2）
- 权重 sha256 = `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2`，与 HuggingFace 的 LFS oid 一致
- **PyPI 上的 `fundus_lesions_toolkit 0.1.0b0` 不可用**：它走 `segmentation_models_pytorch`，权重放在 Google Drive（`www.googleapis.com/drive/v3/files/...`），本机 403 或超时。必须从 GitHub main 分支装（`0.1.1`），那版走 `torchseg` + `huggingface_hub`
- 本机 `huggingface.co` 在 Python 侧 DNS 解析失败，`hf-mirror.com` 可用；`hf-xet` 会绕过 `HF_ENDPOINT` 直连 huggingface.co，必须设 `HF_HUB_DISABLE_XET=1`
- GitHub 直连偶发被重置，走系统代理 `127.0.0.1:7890` 稳定

实现落在 `repro\scripts\generate_masks.py` + `repro\scripts\run_masks.ps1`，坐标系处理与断言的细节写进了 `repro\README.md` 第 6.5、6.6 节。

### 3.3 诚实的不确定性（向用户说明时必须保留）

SMVDR 仓库本身**没有文档化掩膜取值**。EX=1/MA=2/SE=3/HE=4 的约定来自 HACDR-Net issue #6 第三方复现者的选择；HACDR-Net 自身 `DR.py` 的 CLASSES 是 4 类顺序（BG/EX/HE/SE），与其 DDR config 的 `num_classes=5` 不一致。掩膜数值只要与生成模型自洽、且第 4 通道非空，网络可以学到任意编码；但复现 84.01% 这一数字严格依赖与作者一致的编码。

## 4. 上一个会话怎么死的（新会话必须吸取）

13 小时会话累计上下文 35 万 token：`reasoningEffort=max` 导致全程思考文本 306 万字符 + 393 次工具调用 + 每轮注入记忆快照；148 次自动压缩**全部失败**（压缩请求复用主配置，上游报「模型不支持该思考强度」）；客户端按 100 万窗口规划而上游实际硬限约 35 万 → 撞墙，模型输出被压到 1~8 个 token，切换模型后直接 400「内容过长」。

**新会话纪律：**

1. 大文件只分页读（offset/limit），不整读；命令输出过长就重定向到文件再按需读
2. 每完成一个小节就把结论写回本文件 §7 进度日志——这样任何时刻会话报废都能无损重启
3. 一次只做一个子任务，别并行铺开多条调查线
4. 提醒用户把 dsh 的 `reasoningEffort` 从 max 调到 high/medium（这是 148 次压缩失败的直接原因）
5. 已在本文件/手册里的事实，直接引用，不要重新跑命令验证

## 5. 用户偏好

- **不需要 AI 代跑**：用户自己操作，要的是「一步一步照着做」的方案，每条命令带预期输出
- 指令极简（经常只有一个词「继续」）；交付一次到位；反馈是一句话列缺陷+处置决定
- 中文交流

## 6. 下一步行动清单（按序）

1. ~~**装依赖**~~ 已完成（torch 两条 wheel 与 `requirements-segmentation.txt` 全部装好，`requirements-lock.txt` 已冻结，107 个包）
2. **验证环境**（手册第 4 节）：`& $PY repro\scripts\check_setup.py`，含 280 键核对 —— 已实跑通过
3. **放权重**（手册第 5 节）—— 已完成，`weights\best_smvdr.pth` 436,079,327 字节
4. ~~**生成掩膜**~~ 已完成（手册 6.6 节）：`pwsh -File repro\scripts\run_masks.ps1`，96 张掩膜全部生成，444/444 键命中，直方图四类病灶均有像素
5. ~~**组装运行目录**~~ 已完成（手册第 8 节）：`pwsh -File repro\scripts\prepare_data.ps1 -SkipDownload` 组装完毕，目录联接 `repro\EYData_BaseEye_newdata` → `D:\EYData_BaseEye_newdata` 已建
6. ~~**跑通测试**~~ 已完成（手册 9.3、9.4 节）：逐字运行与驱动脚本两条路径都通过，掩膜组 acc 0.6667 / kappa 0.3846，`-NoMask` 控制组 acc 0.5833 / kappa 0.0000，三次退出码都是 0
7. ~~**小批量训练验证**~~ 已完成（手册第 10 节）：批量 1 显存峰值 3155.5 MiB，批量 2 峰值 5333.4 MiB，两项损失、反向、调度器、验证全部通过
8. ~~**逐字运行官方脚本**~~ 已完成（手册 9.4、10.5 节）：`test.py` 与 `train.py` 都改成本机能跑的值，两条逐字运行命令实测退出码都是 0
9. **剩余工作**（本机测试与训练链路已全部验证，剩下三项受数据与硬件限制，见手册第 10.6、11 节）：
   - 完整 MFIDDR 数据需要走官方表单申请：`https://docs.google.com/forms/d/e/1FAIpQLSdQLra-4lEtG6zrASbtPf_Dd5zyEEmQv0HMDgJPBdo97O1qPg/viewform`
   - 为全部 34,452 张图生成掩膜，工具链已就绪（`run_masks.ps1` 会自动遍历样本目录）
   - 论文配置的完整训练（批量 8、100 轮）需要 24 GB 显存，本机 8188 MiB 达不到；本机能跑的是把小批量设置调整后的小规模训练
   - 可选：发邮件向作者索取 HACDR-Net 权重（`xqh51199597@outlook.com`，Qihao Xu），拿到后掩膜保真度可以提升到论文水平

## 7. 进度日志（新会话在此追加）

- 2026-10-09 11:02 上会话因上下文超限死亡，本文件由会话日志分析生成
- 2026-10-09 下午 侦察与方案确定：核实 `D:\EYData_BaseEye_newdata` 实际不存在（上会话记录的「已组装」不成立）；确认替代分割模型可行并已下载权重（138,428,676 字节，sha256 与 HuggingFace LFS oid 一致）；确认 PyPI 版 `fundus_lesions_toolkit 0.1.0b0` 不可用（权重在 Google Drive，本机 403／超时），必须用 GitHub main 分支；确认 `hf-mirror.com` 可用、`huggingface.co` 在 Python 侧 DNS 解析失败、`hf-xet` 需关闭；`pip install --dry-run` 试算完整依赖集合（106 个包）成功
- 2026-10-09 下午 新增 `repro\scripts\generate_masks.py`、`repro\scripts\run_masks.ps1`、`repro\env\requirements-torch.txt`、`repro\env\requirements-segmentation.txt`；改写 `repro\scripts\prepare_data.ps1`（组装项由 4 项扩为 6 项，末尾改为逐张校验）；重写 `repro\README.md` 第 3、6.5、6.6、8.4、9.1、11 节与附录文件表；更新本文件第 1、3、6 节
- 2026-10-09 下午 用户开始手动下载 torch 2.4.1+cu121（2,444,893,595 字节）。依赖安装与后续全部验证等下载结束后进行
- torch 2.4.1+cu121 与 torchvision 0.19.1+cu121 已装好。`requirements-segmentation.txt` 实跑安装成功（exit 0），107 个包，含 `fundus_lesions_toolkit-0.1.1`、`fundus_data_toolkit-0.1.0`、`nntools-0.1.0`、`mmcv-lite-2.2.0`、`mmengine-0.10.7`。`pip freeze` 已写入 `repro\env\requirements-lock.txt`。
- 该文件末尾两行必须是与上游声明逐字相同的裸地址，不能补 `@commit`：`fundus-lesions-toolkit` 的 `pyproject.toml` 声明的依赖是 `fundus-data-toolkit @ git+https://github.com/ClementPla/fundus-data-toolkit` 与 `nntools @ git+https://github.com/ClementPla/NNTools@attrs_dataset`，而 pip 判断两个直接 URL 依赖是否同一个包时比较 URL 字符串本身（`pip\_internal\models\link.py` 的 `links_equivalent` 只忽略 query 与 fragment），补上哈希会成为互相排斥的两个候选并报 `ResolutionImpossible`。`fundus-lesions-toolkit` 那行只在本文件出现，保留 commit 哈希。两个变体都做过 `--dry-run` 对照确认。
- 安装这一步的三个 git clone 需要先设 `$env:HTTP_PROXY` / `$env:HTTPS_PROXY`，git 不读系统代理。
- `check_setup.py` 用两个常量分别核对：`EXPECTED_PARAM_NUMEL = 108990864` 对应 `sum(p.numel() for p in model.parameters())`（274 个可训练张量），`EXPECTED_STATE_DICT_NUMEL = 108994066` 对应 `state_dict()` 的张量元素总数（280 项）。两个数相差 3,202，来自 6 个 BatchNorm buffer：`bn1.running_mean`、`bn1.running_var`、`bn1.num_batches_tracked` 与 `jointLayer.NONLocalBlock2D.W.1` 的同名三项。`repro\README.md` 的 4.2、5.4、10.1、10.3 节与附录文件表已按这两个数同步，5.4 节表头用「元素个数」。
- `check_setup.py` 实跑已全部通过（exit 0），280/280 键命中、形状一致；权重已复制到 `weights\best_smvdr.pth`（436,079,327 字节，data.pkl sha256 `0347526f...fa4d5` 核对一致）；分割工具链的五个导入（`process_image.crop_image_from_gray`、`fundus_data_toolkit.config`、`fundus_data_toolkit.functional`、`fundus_lesions_toolkit.constants.LESIONS`、`fundus_lesions_toolkit.models.hf_hub.HuggingFaceModel`）实测可用。
- 2026-10-09 14:20 掩膜生成完成（`run_masks.ps1`，exit 0）：权重 444/444 键命中、形状一致，96 张掩膜（train/test 各 48 张），类别直方图 0 类 157,123,881 像素、1 类 31,593、2 类 23,281、3 类 23,930、4 类 83,715。
- 2026-10-09 14:2x 组装完成（`prepare_data.ps1 -SkipDownload`，exit 0）：第 2 步预处理 96 张，第 3 步组装出 4 个目录各 48 个文件与两份 CSV，校验段「train 图 48 张，掩膜 48 张，逐张对应」「test 图 48 张，掩膜 48 张，逐张对应」；`repro\EYData_BaseEye_newdata` 目录联接已建。
- 2026-10-09 14:20 测试链路实跑通过（`run_test.ps1 -UseDriver -NumWorkers 0`，exit 0，45 秒）：键名覆盖 280/280、形状 280 个全部一致、显存峰值 759.4 MiB，acc 0.6667 / kappa 0.3846，混淆矩阵 `[[7,0,0,0,0],[2,1,0,0,0],[0,0,0,0,0],[0,0,0,0,2],[0,0,0,0,0]]`。输出文件 `repro\_work\out\smoke_test_result.txt` 与 `smoke_roc_data.csv`。
- 2026-10-09 14:21 无掩膜控制组实跑通过（同命令加 `-NoMask -Name nomask`，exit 0）：acc 0.5833 / kappa 0.0000，全部 12 个样本都被判为 0 类，测试损失 5497.5507（掩膜组 0.7132）。两组都只有 12 只眼，数值没有论文意义，差别来自这 12 个样本的构成。
- 2026-10-09 14:2x 训练链路实跑通过（`run_train.ps1`，exit 0）：批量 1 / 2 个批次显存峰值 3155.5 MiB，默认批量 2 / 4 个批次峰值 5333.4 MiB，两个设置都通过损失有限、梯度存在且有限、调度器学习率变化、验证有限这四项断言。
- 本机剩余能力边界（实测确认）：推理与显存都够用；论文级训练（批量 8、100 轮、完整 34,452 张图）受限于 8188 MiB 显存与数据申请，做不到。
- 2026-10-09 14:26 与 14:27 两次逐字运行 `test.py`（`run_test.ps1` 不加 `-UseDriver`）实测崩溃，日志 `repro\_work\probe\run_test_verbatim.log` 与 `run_test_verbatim2.log`。外部只看到一行 `Stack overflow.`、退出码 3221225725；worker 内的根因是 `ImportError: DLL load failed while importing _flapack: 页面文件太小，无法完成操作。`。成因链：`test.py:198` 写死 `num_workers=24` → Windows spawn 的每个子进程重新执行 `test.py` 顶层 → 导入 `sklearn` → `scipy` → `_flapack`，24 份动态库超出本机剩余提交余量。
- 2026-10-09 15:06 逐字运行 `test.py` 修复后实跑通过：exit 0，acc 0.6667 / kappa 0.3846 / 测试损失 0.7132，耗时 3.95 秒，与驱动脚本模式数值一致。
- 2026-10-09 15:07 驱动脚本测试路径复测通过（`run_test.ps1 -UseDriver -NumWorkers 0 -Name smoke`，exit 0）：键名覆盖 280/280、形状 280 个全部一致、显存峰值 759.4 MiB。
- 2026-10-09 15:07 驱动脚本训练路径复测通过（`run_train.ps1 -UseDriver -BatchSize 2 -Batches 4`，exit 0）：四个批次损失 1.1072、0.9566、0.7767、0.6591，学习率 0.00000916 → 0.00000500，显存峰值 5333.4 MiB。
- 2026-10-09 15:21 逐字运行 `train.py` 实跑通过（`run_train.ps1 -Verbatim`，exit 0）：100 轮、每轮训练 6 个批次与验证 6 个批次，总耗时 822.9 秒，最佳验证 acc 0.7500，末轮验证 loss 0.969 / acc 0.667。产物 `repro\_work\out\best_smvdr.pth` 436,083,734 字节；`weights\best_smvdr.pth` 的官方权重 SHA256 未变（`82043A61...BF24`），仓库根 `final_test_result.txt` 与 `final_roc_data.csv` 的 SHA256 也未变。
- 2026-10-09 15:26 用最终版脚本复跑两条逐字运行命令，结果一致：`run_test.ps1` exit 0（acc 0.6667 / 0.3846 / 损失 0.7132，3.33 秒）；`run_train.ps1 -Verbatim` exit 0，100 轮耗时 845.8 秒，最佳验证 acc 0.7500。四个驱动脚本设置也各复跑一次，退出码都是 0（测试 acc 0.6667 / 0.5833，训练批量 2 峰值 5333.4 MiB、批量 1 峰值 3155.5 MiB）。
- 2026-10-09 15:4x 同步 `repro\README.md`：9.2 节补脚本用法，9.3 节改写为实测报错与内存数值，9.4 节补逐字运行行与耗时区间，10.1 / 10.2 节更新批量与 `num_workers` 取值，10.4 节补 `-Verbatim` 行，新增 10.5 节逐字运行实测记录与 12.1 节 5 处改动表，13 节常见问题整条替换为实测报错，附录文件表更新 `run_test.ps1` 与 `run_train.ps1` 的说明。两个脚本的头部注释同步改写，`run_train.ps1` 新增 `-Verbatim`。四个 `.ps1` 与八个 `.py` 的语法检查全部通过。

## 8. 本机适配的 5 处改动

`test.py` 与 `train.py` 是 git 跟踪文件，按本机的内存与显存改了 5 处数值，其余未动。完整差异用 `git diff -- test.py train.py` 查看。

| 位置 | 原始值 | 本机取值 | 依据 |
| --- | --- | --- | --- |
| `test.py:198` | `num_workers=24` | `0` | 24 个 spawn 子进程重新导入 `sklearn`/`scipy` 超出本机提交余量，见 §7 的 14:26 记录 |
| `train.py:244` | `num_workers=8` | `0` | 同一成因 |
| `train.py:245` | `num_workers=8` | `0` | 同一成因 |
| `train.py:246` | `num_workers=8` | `0` | 同一成因 |
| `train.py:202` | `BATCH_SIZE = 4` | `2` | 实测批量 2 峰值 5333.4 MiB，本机显卡总 8188 MiB，桌面进程另占约 1404 MiB |

`repro\scripts\run_test.ps1` 与 `repro\scripts\run_train.ps1` 分别提供逐字运行与驱动脚本两条路径，逐字运行调用的就是这两个被调整过的文件。恢复原始值只需把上表右列改回左列。
