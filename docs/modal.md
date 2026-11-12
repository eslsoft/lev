# 在 Modal 训练 Lev

本机负责审核、数据冻结和提交；Modal 执行 CUDA 训练与 development 评测。
入口是 `python -m lev.modal_runner`，默认只做本地预检；`--execute` 才创建云端资源并运行。
已有冻结的研究训练数据，但没有独立专家裁定的生产金标准，不能用格式示例绕过预检。已完成 v0.2 真实 GPU 训练与评测，结果见 `docs/model-card-v02.md`。

## 环境与持久化

```sh
uv sync --locked --extra modal
export LEV_KEV_PATH=/Users/saltbo/Develop/oss/kev
```

提交前需要可用的 Modal 身份。Agent 操作遵循 Realmroot 身份发现规则；若不存在对应集成，才使用已授权任务范围内的开发者 Modal 登录。

入口校验现有 Kev checkout 后，将已提交 Git 对象打包成独立快照，不克隆上游、不修改本机 Kev，不复制其本机 Git 配置、凭据、hooks 或未跟踪文件。
Modal 控制进程使用 Python 3.12，Kev 子进程固定 Python 3.13.8、uv 0.9.2 和 Kev 自己的锁文件。任务请求 4 CPU、32 GiB 主机内存（64 GiB 上限）。Modal SDK 固定在 Lev 的可选依赖中，普通 Lev 命令不依赖它。
镜像还包含 Lev 源码；只上传冻结数据 manifest 列出的文件及 manifest 本身。

- `lev-experiments` Volume 挂载到 `/lev-store`，每次使用唯一 `--run-id`，已有目录拒绝复用。
- `lev-model-cache` Volume 挂载到 `/lev-cache`，缓存 Hugging Face 权重。
- `/lev-store/RUN/` 保存数据快照、配置、`modal-job.json` 和 `modal-status.json`。
- `/lev-store/RUN/result/` 保存 Lev 原有 `job.json`、`run.log`、`status.json`、checkpoint 或 native 评测产物。

默认 GPU 是 A100-80GB，作为首轮显存验证起点，不表示已证明需要或足够；可用 `--gpu` 显式指定其他型号。
默认任务超时 3600 秒，可通过 `--timeout` 调整，最大 86400 秒。超时不是费用预算上限，镜像构建等也不包含在这个训练超时里。

## 数据就绪后

先按 `docs/data.md` 完成规则批准和独立复核，冻结四个分区：

```sh
uv run lev validate --data data/curated/en-v1.jsonl
uv run lev prepare --data data/curated/en-v1.jsonl --out artifacts/en-v1

# 本地预检，不连接 Modal
uv run python -m lev.modal_runner benchmark --dataset artifacts/en-v1 --run-id kev-baseline-001

# 原始 Kev development 基线
uv run --extra modal python -m lev.modal_runner benchmark \
  --dataset artifacts/en-v1 --run-id kev-baseline-001 --execute

# LoRA 训练，只有 train 分区传给 Kev
uv run --extra modal python -m lev.modal_runner train \
  --dataset artifacts/en-v1 --run-id lev-train-001 --execute

# 同一 Volume 上的候选模型 development 评测
uv run --extra modal python -m lev.modal_runner benchmark \
  --dataset artifacts/en-v1 --run-id lev-dev-001 \
  --checkpoint /lev-store/lev-train-001/result/checkpoint --execute
```

首次正式训练前，另建包含已复核代表性长样本的小数据版本，保持四个分区和词族隔离，并用独立配置将 epochs 设为 1。运行一次短程训练，查看原始日志里的显存、步速和实际 Modal 用量后再安排完整训练。不要用超时中断来冒充成功的短程训练，也不要用示例数据训练。

单独的 `benchmark` 固定读取 development；完整的校准和 test 通过下文 `experiment` 操作执行，不得把开发集分数当最终测试成绩。

## 失败与结果读取

任务以 detached App 运行，提交客户端等待返回；断开本机后可在 Modal 控制台查任务与 Volume。
在终端查看日志或下载整个实验目录：

```sh
uv run --extra modal python scripts/download_modal_run.py \
  --run-id lev-train-001 --out runs/modal-lev-train-001
```

下载器使用 SDK 逐文件读取并核对长度，避免当前 Modal CLI 目录下载的目录/文件错误。下载后的 `job.json` 仍保留云端绝对路径，不能直接假定本机 `lev report` 能解析它；不要改写原始记录。当前在云端保存的 native 报告和训练日志是原始证据。

自动重试关闭。Kev 当前 LoRA 分支没有完整优化器断点恢复能力；Volume 持久化不等于可断点续训。正常失败会保存状态并提交 Volume，硬终止可能只有后台提交的日志而没有最终状态，此时必须判为未完成。新尝试使用新 run-id，保留旧记录。

## 首批数据仍需完成

1. 明确 IPA/音节采用的词典规范、语素边界、关系定义及词义粒度，并记录实际批准。
2. 提供真实贡献原稿、候选审核记录和来源权限；旧模型输出不直接变成金标准。
3. 逐项保存裁定者、理由与可定位证据，无法确定的记录保留为 uncertain。
4. 按词族分配四个分区，检查各类正反例、长上下文和完整词条覆盖。

不要为了运行云端任务把 draft 改成 approved，也不要伪造独立复核。

## 首轮固定实验入口

`experiment` 操作按固定顺序执行基线、训练、开发集评测、校准、候选测试和基线测试。阈值预先固定 0.5，不用 test 挑模型。每个阶段单独保存状态、原始日志和报告；任何失败停止后续阶段。

```sh
uv run --extra modal python -m lev.modal_runner experiment \
  --dataset artifacts/en-general-002 --run-id lev-general-20261021-01 \
  --timeout 7200 --execute
```

单独的 `benchmark` 仍只读取 development。`experiment` 在 Volume 内运行报告和校准，因此不会遇到下载后绝对路径改变的问题。研究检查点保留 `research_only` 标记，不能描述为已经完成独立人工验收的生产模型。

校准绑定完整检查点文件哈希，不把原机器上的目录路径当成模型身份。下载后同内容的检查点可使用原校准文件，任一文件改变仍会拒绝。原始云端路径保留作为溯源信息，不会被重写。

## 后续迭代：先选模型，再打开测试集

`train` 与 `benchmark` 可分别执行，避免训练结束就自动读取最终测试集。在 development 上冻结选择之后，`finalize` 只执行该检查点的 calibration-read、calibration-fit、test，再评测固定的比较模型，不进行训练或模型选择：

```sh
uv run --extra modal python -m lev.modal_runner finalize \
  --dataset artifacts/en-general-005 --config configs/lev-4b-v02.toml \
  --run-id lev-general005-v02-final-01 \
  --checkpoint /lev-store/lev-general004-v02-train-01/result/checkpoint \
  --compare-checkpoint jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --compare-checkpoint /lev-store/lev-general-20261021-01/result/checkpoint \
  --timeout 3600 --execute
```

比较模型使用各自原生温度，选中模型使用这次独立 calibration 温度；准确率在 0.5 阈值下不受正温度变化影响，Brier 等概率指标应明确说明此差别。`scripts/advance_generalization_cycle.py` 是本次已授权迭代的固定执行脚本：只等待已提交训练，不重复提交；开发集未过预定门槛时保留记录并停止测试阶段，由后续开发迭代处理。

上述运行已经完成，示例记录真实命令；不要复用已有 run-id。004 训练权重保持不变，005 仅修正测试词族隔离，详见 [修正记录](experiments/en-general-005.md)。`advance_generalization_cycle.py` 是已结束的 004 历史执行脚本，不是持续恢复服务。
