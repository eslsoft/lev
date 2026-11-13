# Lev encoder v0.1：ModernBERT-base 英语判断模型

在 Kaggle 免费 T4 上完成真实训练与评分的研究检查点，数据集为 `en-joint-010`，训练 run 为 `enc-train-004`。模型未发布或部署。全部原始产物、失败轮次与负结论均保留。

**与 Lev-4B v0.2–v0.6 不是同一血统。** 不同骨干（ModernBERT-base 149M vs 4B）、不同引擎（`lev.encoder_train` vs Kev）、不同硬件与精度（T4 fp16 vs A100 fp32）。按 `docs/workflow.md` 口径，两边的数字不得放进同一张表做血统内比较；本卡只在并列展示时标注来源差异。

## 输入与输出

输入是冻结包 `en-joint-010` 的一条记录渲染成的文本：`state` 按确定性规则展开，后接问题的 `instructions`。输出是 `false / true` 的未校准概率，`p[1]` 表示问题为真。不做生成、不给解释。

决策阈值 **0.39**，写在 `configs/encoder-modernbert-base-en-joint-010.toml` 的 `threshold` 字段。该字段是在 `enc-train-004` 跑完之后才加入 config schema 的，所以**那次 run 的 `job.json` 里没有它**：`config_sha256 = b16eb329…` 正是加字段前的 config 摘要。

为把阈值真正写进 run，已用保存的 checkpoint 重新评了一次分：**`enc-dev-005`**（产物在 `runs/kaggle-enc-dev-005/`）的 `job.json` 记录 `config.threshold = 0.39`、`config_sha256 = 46f335f8…`，`operation = benchmark`，`status = succeeded`。重跑分数与 `enc-dev-004` 完全一致（阈值 0.5 → 212/303，0.39 → 222/303，逐类也相同），说明重跑没有改变结论。该 run 自带的 `lev-report-t0.39.json`（在 Kaggle 上由 `lev report` 直接生成）为 222/303 = 73.27%，completed 303 / missing 0。

0.39 的报告已落盘为 `runs/kaggle-enc-train-004/enc-dev-004/lev-report-t0.39.json`（**222/303 = 73.27%**，semantic_relation 从 76/156 升到 86/156），由 `lev.report.summarize` 同一函数生成。Kaggle 产出的原始报告是阈值 0.5 的 212/303 = 69.97%。本地重跑评分受阻于 transformers 版本错配（Kaggle 5.x 写入 `tokenizer_class: TokenizersBackend`，本地固定 4.57.6），失败产物保留在 `runs/enc-dev-004-local-failed-01/`。

## 主结果：development 147 判断

| 轮次 | 数据包 | train 判断 | dev 147 | passage_correction | contextual_sense |
|---|---|---:|---:|---:|---:|
| `enc-train-001` | `en-joint-002` | 15,515 | 133/147 = 90.48% | 128/139 = 92.09% | 5/8 = 62.50% |
| `enc-train-003` | `en-joint-006` | 80,363 | 80/147 = 54.42% | 77/139 = 55.40% | 3/8 = 37.50% |
| **`enc-train-004`** | **`en-joint-010`** | **19,153** | **136/147 = 92.52%** | **131/139 = 94.24%** | **5/8 = 62.50%** |

`enc-train-004` 在 303 条多任务 development 上为 212/303 = 69.97%（阈值 0.39 时 222/303 = 73.27%），`brier_completed_only`（passage_correction）0.05576，`dataset_sha256` = `fc7c843a7936d8cc9cfdcb18091d06a71faabbad26aad72f3374177b4b6a30c9`，与本地 dry-run 逐字节一致。

阈值 0.39 由 development 上的网格搜索选出，拟合在 303 条样本上，增益不保证泛化。**因此 222/303 是样本内数字**（阈值就是在这批样本上挑的，有乐观偏差）；未经拟合的数字是阈值 0.5 的 212/303 = 69.97%。它对 test 而言才是预注册：test split 未读取。

## 负结论：`en-joint-006` 是退化解

`en-joint-006` 把 `semantic_relation` 从 804 扩到 65,652 条，占 train 的 **81.7%**，标签平衡 0.501，任务是「从两段 WordNet gloss 判 hypernymy」。3 epochs / lr 3e-5 学不动：

- train loss 0.6995 → 0.6949 → **0.6937**（ln 2 = 0.6931）；
- development **147/147** 条预测落在 0.5 ± 0.05；
- label=0 的 p 均值 0.4998，label=1 的 p 均值 0.5001。

它同时把 `passage_correction` 从 92.09% 淹到 55.40%。数据更多、家族交叉为零、词覆盖 55.0%，模型却更差——**词覆盖度量的是数据形状，不是模型能力**。

## `semantic_relation` 至今没学会

`enc-train-004` 恢复并略超旧基线，但新增的 semantic_relation 判断只有 **76/156 = 48.72%**，即抛硬币。那 65k 条原料对模型零贡献，只是不再毒害其它任务。**不得把它算作多任务能力提升。**

## 训练成本（实测，非估算）

| 数据包 | train 判断 | 每 epoch | 3 epochs | 峰值显存 |
|---|---:|---:|---:|---:|
| `en-joint-010` | 19,153 | 406–407 秒 | 1,220 秒（20.3 分钟） | 6.90 GiB |
| `en-joint-006` | 80,363 | 1,352–1,357 秒 | 4,065 秒（67.7 分钟） | 6.85 GiB |

单卡 T4 16 GiB，fp16，`max_len = 3072`（两包均不截断）。实测吞吐 `en-joint-010` 为 47.1 判断/秒（19,153 / 406.77 秒），`en-joint-006` 为 59.3 判断/秒（80,363 / 1,354.83 秒）。`max_len` 是拒绝线，`max_tokens = 8192` 是显存预算；超长输入直接失败，不截断。

## 评估包 `en-eval-001`

多任务评估包，development 4,899 判断，覆盖 7 类：grammaticality 3,094、pronunciation 672、syntax 544、lexical_analysis 286、semantic_relation 156、passage_correction 139、contextual_sense 8。**尚未训练评分。**

- `audit_family_isolation.py` 报 **44 个家族违规**（syntax 34、lexical_analysis 10、semantic_relation 4、pronunciation 4、grammaticality 3、contextual_sense 1），其中 4 个在 `en-joint-010` 本身就存在；`prepare()` 只校验字面 `family_id`（零交叉），更严的词族根视角见 `data/research/en-eval-001/isolation-report.json`。
- 重叠不可避：base train 的 passages 覆盖极宽词汇面，任何新 held-out 句子几乎必然与某条 train passage 同词族。新增项按「家族未被 base 分配」筛选（UD 排除 823、BLiMP 排除 2,881、CMUdict 排除 34、Wiktionary 排除 22），仍挡不住 `attach()` 事后把文档与词连起来。
- 标签全部为 `source_derived` / `source_labeled`，**不是独立裁决**。

## 已知缺口（未覆盖，不声称覆盖）

| 缺口 | 状态 |
|---|---|
| collocations | 需 COCA/BNC 级搭配数据，付费许可与零成本约束冲突 |
| register | 同上 |
| etymology | 本地无词源原料 |
| pronunciation | 8,930 / 35,000 词（25.5%）无发音来源 |

`pragmatics` 只出现在 test split，train/dev/cal 为零，无法训练也无法在 development 上度量。

## 不可比与未验证

- 无温度校准；`brier_completed_only` 是未校准概率，不能与 v0.2 的校准后 Brier 并列。
- 不产生 `calibration.json`，没有 `serve` 入口。
- ModernBERT-large（395M）未跑；DDP 未接，T4×2 的第二张卡用不到。
- test split 未读取。

## 复现

```sh
# 训练（Kaggle T4，fp16）
uv run --extra encoder python -m lev encode \
  --dataset artifacts/en-joint-010 \
  --config configs/encoder-modernbert-base-en-joint-010.toml \
  --out runs/enc-train-004 --device cuda --execute

# development 评分
uv run --extra encoder python -m lev encode-benchmark \
  --dataset artifacts/en-joint-010 \
  --config configs/encoder-modernbert-base-en-joint-010.toml \
  --checkpoint runs/enc-train-004/checkpoint \
  --split development --out runs/enc-dev-004 --device cuda --execute

uv run lev report --benchmark runs/enc-dev-004 --threshold 0.39 \
  --out runs/enc-dev-004/lev-report.json
```

数据包构建：`scripts/build_coverage_corpus.py --version en-joint-010 --relation-records 2000`。评估包构建：`scripts/build_eval_bundle.py`。
