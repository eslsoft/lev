# Lev

Lev 是面向语言知识的专用判断模型项目。当前研究模型 **`lev-4b-v0.2`** 从固定版本的 Kev-4B 继续训练。Kev 提供模型、LoRA 训练和推理；Lev 管理审核标准、可追溯标签、数据分区、实验记录与语言任务评测。

**当前状态：已在 Modal 完成 v0.2 研究版训练、独立校准和修正后的留出评测，检查点已下载；尚无经过独立裁定的生产金标准，也未发布或部署模型。** `data/examples` 中的数据仅演示格式，不代表语言学金标准或模型成绩。第一阶段限定英语、真假判断；输入是开放的自然语言审核问题，不限于固定审核类别。其他语言和“不确定”类别需要单独设计与验证。

当前冻结研究数据为 `en-general-005`：4,309 条请求、6,478 个判断，其中训练 4,424 个判断。同一修正后的 932 题测试：Kev **66.63%**、Lev v0.1 **66.42%**、Lev v0.2 **84.87%**。语法仍仅 66.25%；旧直接问法开发集从 94.61% 小幅回退至 93.63%。一次词族隔离修正发生在初次测试之后，不能将修正评估视为独立复现。权重、校准、完整结果和限制见 [v0.2 模型卡](docs/model-card-v02.md)；保留 [v0.1 历史模型卡](docs/model-card.md)。

## 持续训练进行中

最终模型目标保持 active。v0.3已完成真实训练和开发比较：新开发提高，但旧能力回退超过预定门槛，因此没有进入新测试。v0.4正在Modal用同一干净数据训练两轮（每轮6,505请求／8,115判断），后续开发、校准和35,000词评测按固定规则接续。另已冻结1,551个目标词的独立WiC语境补充，尚未模型评分。真实证据、限制和恢复入口见 [持续训练](docs/continuous-training.md)；训练或数据构建完成不代表最终能力达标。

## 35,000 词测试目标

已建立 [美国英语 35,000 词研究测试集](docs/vocabulary-benchmark.md)：35,000 个唯一词头、368,056 个判断，按词统计全部已排定测试是否通过，保留每词覆盖缺口和既有训练接触。**该集尚未运行模型；上面的 v0.2 成绩不代表这一词汇覆盖水平。**

## 新增真实语境与篇章纠错数据

已冻结 [en-context-002 补充研究包](docs/experiments/en-context-002.md)：训练 3,721、开发 147、校准 118、新留出 159 个判断，共 4,145 个。主体是有人工修改依据的真实写作纠错；词义部分仅有 48 个显式目标单词（训练 17 个），不能算已补齐 35,000 词的语境能力。严格保留原词汇基准与历史留出隔离，全部来源／分区／导出审计通过。**仅限非商业研究，尚未训练。**

## 本地开始（不下载模型）

需要 Python 3.12/3.13、uv，以及现有 Kev 仓库。不要安装 PyPI 上同名的 `kev` 包。

```sh
uv sync --locked
export LEV_KEV_PATH=/Users/saltbo/Develop/oss/kev
uv run lev check-upstream
uv run lev validate --data data/examples/format-only.jsonl
uv run lev prepare --data data/examples/format-only.jsonl --examples --out artifacts/format-demo
uv run python -m unittest discover -s tests -v
```

`LEV_KEV_PATH` 只在本机环境设置，不硬编码进项目配置。项目不会克隆或修改 Kev 源码。训练/推理使用 Kev 自己的 `uv.lock` 和环境；日常数据整理、测试不需要 PyTorch 或 GPU。

## 项目结构

```text
configs/lev-4b.toml       模型名、上游提交、基础模型与检查点的固定版本、训练参数
configs/lev-4b-v02.toml   当前 v0.2 实验配置（需显式选择）
rubrics/en-v1.json       历史八类规则初稿，不限制当前开放问题接口
src/lev/dataset.py       标签校验、词族隔离、导出 Kev JSONL、冻结及哈希检查
src/lev/workflow.py      训练、基准、校准、服务命令及执行记录
src/lev/bridge.py        Kev 环境里的校准和 Lev 命名适配
src/lev/report.py        单项、分类、完整词条和未完成判断的统计
src/lev/upstream.py      固定提交和上游命令契约检查
data/examples/         格式示例，禁止用于正式训练
data/curated/          后续经复核的正式数据
 docs/                  标注规范、实验步骤、当前限制
 tests/                 数据与评测行为、上游源码契约测试
artifacts/              本地冻结数据，Git 忽略
runs/                   本地日志、模型和评测产物，Git 忽略
```

## 数据先行

参见 [数据与标注约定](docs/data.md)。一个记录包含共享上下文及若干判断问题；理由、证据、裁判身份独立保存，不进入模型输入。同词族及同词条变体不得跨分区。

四个分区：`train` 训练，`development` 调整规则/参数/阈值，`calibration` 校准概率，`test` 最终评估。所有分区明确分配后统一导出，不依赖逐行随机切分。

完成审核规范并将其状态改为 `approved`，准备具有 `adjudicated` 标签的正式数据后：

```sh
uv run lev validate --data data/curated/en-v1.jsonl
uv run lev prepare --data data/curated/en-v1.jsonl --out artifacts/en-v1
```

未裁定数据、缺失标签、示例数据、同词族跨分区、重复 ID 或复制内容跨分区会阻止正式导出。导出不静默丢弃错误记录。已有产物目录不覆盖。

## Modal GPU 训练

云端入口已接入训练与 development 评测，默认本地预检，显式 `--execute` 才运行。依赖安装、持久化产物、提交命令和当前限制见 [Modal 工作流](docs/modal.md)。正式数据仍需先完成审核与冻结；已完成真实 GPU 训练、检查点重新加载、校准与留出评测，结果及限制见 [v0.2 模型卡](docs/model-card-v02.md)。

## 模型工作流

以下为使用默认 v0.1 配置的通用流程示例；实际 v0.2 权重、校准路径与显式配置命令见 [v0.2 模型卡](docs/model-card-v02.md)。

所有昂贵操作默认只输出计划；加 `--execute` 才会下载依赖/权重并运行。这个开关是 CLI 的 dry-run 约定。默认训练设备 CUDA；Mac 可显式指定 `--device mps`（Kev 训练使用 fp32，显存/统一内存另行实测）。

```sh
# 先测固定 Kev 基线
uv run lev benchmark --dataset artifacts/en-v1 \
  --checkpoint jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --split development --out runs/kev-baseline --execute

# 从 Kev 继续训练，只读 train.jsonl
uv run lev train --dataset artifacts/en-v1 --out runs/lev-4b-001 --execute

# 在 development 比较新模型，所有变更完成后冻结候选
uv run lev benchmark --dataset artifacts/en-v1 --checkpoint runs/lev-4b-001/checkpoint \
  --split development --out runs/lev-dev-001 --execute
uv run lev report --benchmark runs/lev-dev-001 --threshold 0.8 --out runs/lev-dev-001/lev-report.json

# 只在 calibration 拟合温度；原始检查点不改写
uv run lev benchmark --dataset artifacts/en-v1 --checkpoint runs/lev-4b-001/checkpoint \
  --split calibration --out runs/lev-cal-read-001 --execute
uv run lev calibrate --benchmark runs/lev-cal-read-001 --out runs/lev-cal-001 --execute

# 冻结温度和阈值后测 test，不能据此循环调参
uv run lev benchmark --dataset artifacts/en-v1 --checkpoint runs/lev-4b-001/checkpoint \
  --calibration runs/lev-cal-001/calibration.json --split test --out runs/lev-test-001 --execute
uv run lev report --benchmark runs/lev-test-001 --threshold 0.8 --out runs/lev-test-001/lev-report.json

# 本地研究服务：显式 Lev 模型名，仅绑定 127.0.0.1
uv run lev serve --checkpoint runs/lev-4b-001/checkpoint \
  --calibration runs/lev-cal-001/calibration.json --out runs/lev-serve-001 --execute
```

`0.8` 是初始决策阈值，不是 80% 准确率保证。应在 development 上确定并冻结阈值，再进行最终测试。温度校准不提高 argmax 正确率；它调整概率尺度。拟合指标也不是独立测试成绩。

服务沿用 `POST /v1/systemone`，请求须显式携带 `"model":"lev-4b-v0.1"`；`GET /v1/models` 展示 Lev 名称。推理内核仍来自 Kev，保留上游归属。没有自动部署、上传或发布模型。

详细步骤和限制见 [实验工作流](docs/workflow.md)、[历史 v0.1 模型卡](docs/model-card.md)。

## 上游与复现

- Kev 本地提交：`1f04cfa28e6a708485b28a9585010b9e436b776e`。
- Kev-4B 检查点：`139fdd94f1b6a6ad80cc15e08fcb99cac885a101`。
- Qwen3.5-4B-Base：`1001bb4d826a52d1f399e183466143f4da7b741b`。
- [Kev 源码](https://github.com/jaredpalmer/kev)、[Kev-4B 模型卡](https://huggingface.co/jaredpalmer/kev-4b)。上游代码/权重按其 Apache-2.0 条款使用；后续数据来源单独记录权利和使用范围。

固定版本不等于跨硬件逐位一致。每次实验保留命令、代码/数据哈希、原始日志和 Kev 原生指标；更换硬件、精度或内核后重新验证。
