# Search-R1-style Qwen3-4B Search Agent

面向单 GPU 的多轮检索问答后训练项目：**规则轨迹 SFT → unmerged GRPO → 固定验证集评测**。基于 Qwen3-4B-Instruct-2507，使用 JSON 搜索/回答协议和本地 BM25 检索。

这是 Search-R1 思路的工程适配，**不是原论文复现，也不是官方 HotpotQA 榜单协议**。代码支持云端训练；仓库不包含真实数据、模型权重或云端原始实验报告。

## 功能

- SFT：完整教师轨迹回放、按问题留出 val loss、生成 probe、LoRA、梯度检查点。
- GRPO：多问题组采样、有效组筛选、证据/协议辅助奖励、冻结 SFT 参考 adapter、可恢复检查点。
- 多问题并行 rollout 和批量评测；检索返回 token 不参与策略损失。
- TensorBoard 训练指标与 GPU 遥测；可选每次只显示五个终端指标。
- 数据/配置/代码/权重身份校验，避免混用实验数据或检查点。
- 独立300题评测包，以及从原教师轨迹与GRPO题库补齐300条规则SFT轨迹的入口。

## 实验记录

用户提供的同一300题验证集结果（2026-09-21；原始云端JSON未随仓库上传）：

| 模型 | 总体 EM | 原50题 EM | 新250题 EM |
|---|---:|---:|---:|
| SFT r32 / 3 epoch | 0.5433 | 0.6000 | 0.5320 |
| GRPO step5 | 0.5600 | 0.6400 | 0.5440 |
| GRPO step10 | 0.5467 | 0.5800 | 0.5400 |
| GRPO step15 | 0.5667 | 0.6000 | 0.5600 |

原SFT使用183条轨迹（165训练、18内部验证）；300条SFT为后续实验入口，**尚无已验证效果结果**。GRPO step15净增7道正确答案，不据此宣称统计显著提升。小规模共享语料结果不能与原论文Wikipedia检索结果直接比较。

## 文件导航

| 位置 | 职责 |
|---|---|
| `search_train.py` | 云端训练、评测和检查点管理 |
| `search_task.py` | JSON动作、BM25、轨迹执行与数据校验 |
| `grpo_sampling.py` | 问题组采样、奖励信号筛选 |
| `training_common.py` / `training_visualization.py` | 训练计算与遥测 |
| `prepare_*_experiment.py` | 从已有配置派生新的实验配置 |
| `prepare_search_data.py` / `expand_search_data.py` | 官方数据转换、扩展GRPO题库 |
| `scripts/` | 训练显示、检查点评测、300题/轨迹准备、云端环境及打包入口 |
| `search/` | 示例/历史配置和虚构demo；真实数据本地生成 |
| `tests/` | CPU契约测试与模拟训练测试 |
| `docs/` | [文档导航](docs/README.md)、操作说明及历史设计 |
| `dist/` / `outputs/` / `reports/` | 本地生成产物，不上传Git |

保留根目录Python模块布局，因为检查点身份包含这些文件的哈希；不要随意移动或修改后尝试恢复旧训练。

## 本地检查

Python 3.12。以下基础检查不加载模型：

```bash
python search_train.py validate
python -m unittest discover -s tests -p test_sft300.py -v
python -m unittest discover -s tests -p test_compact_console.py -v
```

完整测试需要torch等相应依赖：

```bash
python -m unittest discover -s tests -v
python verify_local.py
python build_package.py
```

虚构demo仅用于软件检查，不代表真实训练效果。GPU依赖见 `requirements-train.txt`，不要在CPU机器上盲目安装云端GPU环境。

## 云端使用

克隆只获取代码；需另外准备模型、HotpotQA数据包及对应实验配置。已有云端项目请先保留原配置、模型与数据，不能用仓库里的demo配置代替正式实验。

- 初次准备：[基础云端手册](docs/plans/search-r1-runbook.md)
- 当前多问题GRPO：[运行说明](docs/plans/2026-09-21-cross-question-runbook.md)
- 300题验证集：[评测说明](docs/plans/2026-09-21-validation300.md)
- 300条SFT（最多3次搜索）：[训练说明](docs/plans/2026-09-21-sft300.md)

精简终端日志时，将 `python -u search_train.py` 替换为 `python -u scripts/train_compact.py`，其余参数保持一致。完整TensorBoard及文件日志仍保留。

## 方法与数据边界

动作仅为 `{"search":"query"}` 或 `{"answer":"short answer"}`。规则教师使用训练标注的支持标题作为搜索词、标准答案作为终点；支持文档命中检查不等于语义充分性证明。SFT内部holdout与任务验证集含义不同。GRPO只对模型生成token计算策略损失。

## 来源与归属

- [Search-R1论文](https://arxiv.org/abs/2503.09516) / [官方实现](https://github.com/PeterGriffinJin/Search-R1)：方法参考。
- [HotpotQA](https://hotpotqa.github.io/)：数据为 CC BY-SA 4.0，Yang et al., EMNLP 2018；不在本仓库重新分发真实数据。
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`，模型权重不随仓库发布。
- `search/demo` 为项目自编虚构fixture，manifest标注CC0-1.0。

第三方模型、数据及依赖各自遵循其许可证。本次整理未另行指定项目代码许可证。
