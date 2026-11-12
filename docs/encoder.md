# 编码器决策模型（Plan A）

这是一条与 Kev-4B 并行的训练路径：**ModernBERT 骨干 + 二维 true/false 头**。它读同一份冻结数据包、产出同一套证据、由同一个 `lev report` 评分，但不调用 Kev。

## 为什么需要这条路径

Kev 路径把 4B 骨干以 fp32 冻结权重常驻显存：

| 组成 | 占用 |
|---|---:|
| Qwen3.5-4B 冻结权重（fp32） | ~14.9 GiB |
| LoRA 33.8M + AdamW 动量 + 梯度 | ~0.5 GiB |
| 指针头与激活（已开 checkpointing） | 其余 |

实测峰值 20,044,277,760 bytes（`lev-joint001-v03-train-01`，A100-80GB），且 **32 条数据的 smoke 与 6,505 条的全量训练峰值几乎相同**——这 20 GB 与数据量无关，是固定开销。16 GiB 的单卡（T4/P100）连权重都装不下。

而冻结包的实际形态是短文本：

| 数据包 | state 最长 tokens | 说明 |
|---|---:|---|
| `en-joint-002` train | 2,329 | 13,905 请求 / 15,515 判断，中位 198 字符 |
| `en-us-vocab-35000-v3` | 82–84 | 单词级判断 |

所以 `max_state = 7552` 是过配。一个 1.5 亿参数的编码器能装进 16 GiB，并且把"判真假"这件事做得一样好——它本来就是一个判别任务，不需要生成能力。

## 它是什么

- 骨干：`answerdotai/ModernBERT-base`（149,655,232 参数）或 `-large`（395,881,664），原生 8192 上下文，按 commit SHA 固定。
- 头：`AutoModelForSequenceClassification(num_labels=2)`，`p[1]` 是 `true`。
- 输入：`state` 渲染成确定性文本 + `"\n\n"` + `question.instructions`（其中已含 claim）。这与 Kev `api_request` 的投影一致：**只有 state 与问题文本，没有 label、review、evidence、`_meta`**。
- 输出：二维 softmax，写出 Kev 原生 `rows.json` 格式，因此 `report.align` 与 `report.summarize` 不需要任何改动。

`state` 是对象时按键名排序渲染为 `key: value` 行；是字符串时原样使用。

## 命令

```sh
# 只打印计划，不下载权重、不用 GPU
uv run --extra encoder python -m lev encode \
  --dataset artifacts/en-joint-002 --out runs/enc-train-001

# 真实训练
uv run --extra encoder python -m lev encode \
  --dataset artifacts/en-joint-002 --config configs/encoder-modernbert-base.toml \
  --out runs/enc-train-001 --device cuda --execute

# development 评测（只读 development）
uv run --extra encoder python -m lev encode-benchmark \
  --dataset artifacts/en-joint-002 --config configs/encoder-modernbert-base.toml \
  --checkpoint runs/enc-train-001/checkpoint --split development \
  --out runs/enc-dev-001 --device cuda --execute

# 评分：与 Kev 路径完全相同的入口
uv run lev report --benchmark runs/enc-dev-001 --threshold 0.5 \
  --out runs/enc-dev-001/lev-report.json
```

默认 Lev 环境仍然 `dependencies = []`；`encoder` extra 才装 `torch` 与 `transformers`。

## 不截断

超过 `max_len` 的输入**直接失败**，不截断：

```
ValueError: 3 of 15515 inputs exceed --max-len 3072 (abc/q0=3180, ...); raise --max-len rather than truncate
```

截断会静默改变被判断的命题，而报告看不出来。`en-joint-002` train 最长 2,428 tokens，所以 `max_len = 3072` 留了余量；换数据包要先看 `runs/*-tokenization.json` 的 `max_packed_tokens`。

## 与 Kev 路径的关系

相同：冻结数据集目录、`job.json` / `status.json` / `run.log` 证据契约、`native/rows.json` 预测格式、`lev report` 的全部指标口径（含 missing 不当作判对、whole_cases 只算 complete）。

不同：不调用 Kev、不需要 `LEV_KEV_PATH`、**不产生校准**（`calibration_sha256` 固定为 `null`）、**不提供 `serve`**。`job.json` 用 `engine: "encoder"` 记录实际引擎，`report` 会把它带进结果。

## 显存与时间（T4 级单卡，估算）

| 骨干 | 参数 | fp32 权重+梯度+AdamW | 每 epoch |
|---|---:|---:|---:|
| ModernBERT-base | 149M | ~2.7 GiB | 5–20 分钟 |
| ModernBERT-large | 395M | ~7 GiB | 13–50 分钟 |

这些是估算，不是实测。真实吞吐必须先跑一次短程训练，看 `checkpoint/lev-encoder.json` 里 `history[].seconds` 与 `history[].peak_device_bytes`。

精度：`dtype` 只影响前向 autocast，权重始终 fp32。T4（sm_75）与 P100（sm_60）没有 bf16 算术单元，`dtype = "bf16"` 在它们上面会退化或报错；需要 bf16 就用 Ampere 及以上的卡。

## Kaggle

`scripts/make_kaggle_notebook.py` 生成可直接上传的 notebook：

```sh
uv run python scripts/make_kaggle_notebook.py --out notebooks/kaggle-encoder-smoke.ipynb
```

操作要点：

1. Settings → Internet 打开，否则按 SHA 拉不到骨干权重。
2. Settings → Accelerator 选 GPU T4 x2 或 P100。这条路径是单设备，第二张卡不会被用到。
3. 冻结包要作为 Kaggle Dataset 上传：`cd artifacts && zip -qr en-joint-002.zip en-joint-002`，再把 notebook 里的 `DATASET` 指向解压目录。
4. `/kaggle/working` 只有 Save Version 才持久化。每轮实验单独 commit，不要串在一个 session 里。
5. 免费额度约 30 GPU 小时/周（按需浮动，以 Settings 页为准），单 session 9–12 小时上限。上面所有单轮都在 1 小时内，超时风险主要来自把多轮实验串在一起。

## 产生的证据

```text
runs/enc-train-001/
  job.json              命令、配置哈希、数据 manifest 哈希、Lev 源码哈希
  run.log               encoder_train 原始 stdout/stderr
  status.json           成功/失败与退出码；中断也记录失败
  checkpoint/           HF 权重 + tokenizer
    lev-encoder.json    骨干固定版本、max_len、每 epoch loss/耗时/峰值显存
runs/enc-dev-001/
  job.json              含 checkpoint 逐文件哈希
  run.log
  status.json
  native/rows.json      Kev 格式预测
  lev-report.json       lev report 输出
```

输出目录必须不存在，不复用、不覆盖。失败原样保留，不重试。

## 分批：按长度装箱，不按全包最长补齐

`--max-tokens` 是每批**补齐后 token 数**的上限，与 `--batch` 同时生效。批次按长度排序装箱，所以一个 2,400 token 的记录不会把同批的 200 token 记录一起补齐到 2,400。

这不只是省时间。SDPA 的数学核会把注意力矩阵物化成 `L x L`，2,400 token 的补齐比 400 token 贵约 36 倍。本机实测（Apple MPS，fp32，无 flash 核）：

- 不装箱、`batch = 4`、补齐到 2,429 token：**26.9 GiB，OOM**（上限 27.20 GiB）。
- 装箱后同一份数据变成 9 批，CPU fp32 一轮 71.85 秒完成。

所以 `max_tokens` 与 `max_len` 是两件事：`max_len` 是拒绝线，`max_tokens` 是显存预算。16 GiB 卡上不要指望 fp32 + 无 flash 核能跑 `max_len = 3072`。

## 实测（本机 CPU，流水线 smoke）

`artifacts/en-joint-001-smoke`（train 32 请求 / development 147 判断），1 epoch，fp32，CPU：

```text
train: 32 pairs in 9 batches, 1 epochs, batch 4 / 8192 padded tokens, max_len 3072
epoch {"epoch": 1, "pairs": 32, "seconds": 71.85, "train_loss": 1.698, "peak_device_bytes": null}
```

`encode` → `encode-benchmark --split development` → `lev report` 三步都跑通了，`lev-report.json` 里 `engine` 为 `encoder`、`calibration_sha256` 为 `null`、checkpoint 按文件逐个哈希。

**这不是模型成绩**。32 条数据训 9 步的模型把 139 条 development 判断全部接受：70/139 = 50.36%，`false_acceptances` 69、`false_rejections` 0。该 split 的基率正好是 70 真 / 69 假，所以"全判真"就是 50.4%——它只证明流水线与评分口径接通了，没有证明任何关于模型能力的事。

## 尚未验证

- 本仓库**还没有**这条路径在真实数据量上的模型分数。上面的显存与时间是估算加一次 smoke，不是成绩。
- 没有温度校准。`report` 的 `brier_completed_only` 用的是未校准概率，不能与 v0.2 的校准后 Brier 并列。
- 与 v0.2–v0.6 血统**不可比**：不同骨干、不同精度路径、不同硬件。按 `docs/workflow.md` 的口径，换骨干或换硬件后要重新验证，不能把两边的数字放进同一张表。
- 编码器不产生 `calibration.json`，也没有 `serve` 入口。
- T4 上的真实吞吐与显存未测。`dtype = "fp16"` 加 GradScaler 的路径也没有在真实 GPU 上跑过。
