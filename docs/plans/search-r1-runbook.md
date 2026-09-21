# Qwen3-4B Search-R1 风格适配：云端手册

本项目仅包含搜索任务，模型为 Qwen/Qwen3-4B-Instruct-2507，固定提交 cdbee75f17c01a7cc42f958dc650907174af0554。本地只准备代码与测试，模型执行仍限定 Linux/CUDA。

## 实现与限制

- 方法：搜索轨迹 SFT → 可选 CPU 合并（评测 B1-init 用）→ **推荐** `--unmerged-sft` 在 SFT adapter 上继续 GRPO → 分别加载评测。
- 动作协议：每轮输出 `{"search":"查询"}` 或 `{"answer":"简短答案"}`。不添加思考标签；这是非思考指令模型的适配协议，不是原论文标签协议的逐行复现。
- 模型调用后，CPU BM25 从共享语料返回文档。每个动作的完整上下文保留，只有当次 assistant 输出及 EOS 参与损失；历史动作和搜索结果只作条件。
- GRPO：组内对最终 EM 奖励做标准化；温度 1、无 top-p/top-k 截断。每次新鲜采样对应一次更新；未合并模式下 KL 参考为冻结的 SFT adapter。没有对“多搜”加奖励。
- 后端串行，优先验证正确性。bf16 必需。中途终止、非法动作或超搜索上限，最终奖励为 0。上下文过长不静默裁剪。
- EM 使用 Unicode NFKC、小写与空白归一化，不是 HotpotQA 官方标点评分器；报告不能与官方榜单直接比较。
- GPU 训练、真实 tokenizer、保存重载和中断恢复以云端实测为准。

## 上传与环境

把 dist/search-r1-qwen3-4b.zip 和同名 .sha256 上传至数据盘（例如 /root/autodl-tmp），使用新空目录：

```bash
cd /root/autodl-tmp
sha256sum -c search-r1-qwen3-4b.zip.sha256
mkdir search-r1-qwen3-4b
unzip search-r1-qwen3-4b.zip -d search-r1-qwen3-4b
cd search-r1-qwen3-4b
export POSTTRAIN_DATA_ROOT=/root/autodl-tmp
bash scripts/setup_6000d.sh
source .venv/bin/activate
source scripts/env_6000d.sh
python verify_package.py
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -m pip check
python hardware_probe.py
```

HF-Mirror 是可选第三方下载源；可直连官方时不必设置。若服务器上已有同一 revision 的基础模型（含 `origin.json` 且权重摘要可过校验），可软链接到 `models/Qwen3-4B-Instruct-2507` 以复用磁盘，**不要**链接旧 outputs/合并模型。

## 先运行微型流程测试

```bash
python search_train.py validate
bash scripts/run_search.sh smoke
```

smoke 使用6篇虚构文档，每个划分1个问题，运行1步 SFT、合并、1步 GRPO 和 B3 重载评测。它只用于发现依赖、tokenizer、显存和训练接口问题；不能作为实验成绩。组内奖励可能全0/全1，有效更新门槛可能不通过，这是需要观察的诊断，不代表性能提升。不要重复运行到已有输出目录；失败后根据报错恢复检查点，或给新实验使用新的配置输出路径。

## 正式数据：HotpotQA 小规模方法验证

从官方 https://hotpotqa.github.io/ 下载 Training set 和 Dev set (distractor)，放在云端数据盘，例如 search/raw/ 下。训练文件约535MB、开发文件约44MB。数据许可 CC BY-SA 4.0，保留 Yang et al., EMNLP 2018 来源。转换器不联网、不自动下载。

```bash
python prepare_search_data.py --train search/raw/hotpot_train_v1.1.json --dev search/raw/hotpot_dev_distractor_v1.json
python search_train.py validate --config search/pilot.json
```

默认选择200条训练候选、50条验证、100条留出测试；实际训练数可能因教师检索失败减少。项目 test 来自官方有标签 dev，并非官方无标签 test；项目 val 来自官方 train。共享语料汇总所有选中问题的上下文及干扰文档，检索器看不到问题的 gold/support_ids。这是受控共享语料实验，不是完整 Wikipedia 检索。

SFT 教师轨迹用训练集 supporting titles 生成查询，然后输出训练答案；只验证支持文档能被命中，不保证截断后信息充分。正式训练前人工抽查；不能称为教师模型推理轨迹或证据验证完备的数据。严格归一化问题去重不排除近义问题或预训练污染。

## 正式执行

```bash
bash scripts/run_search.sh prepare
bash scripts/run_search.sh sft
bash scripts/run_search.sh grpo
python search_compare.py reports/search/B0-adaptive-val.json reports/search/B1-adaptive-val.json reports/search/B1-init-adaptive-val.json reports/search/B3-adaptive-val.json
```

prepare 验证数据、准备模型并评测无检索/固定检索/自适应检索基线。SFT 默认32步、GRPO默认20步仅为小规模初始预算，不代表收敛。先看 val 决定参数，固定后再运行 `bash scripts/run_search.sh final`。所有指标报告拒绝覆盖；需要新实验时使用独立目录和配置。

TensorBoard：`tensorboard --logdir outputs --host 127.0.0.1 --port 6006`，沿用原手册 SSH 转发。记录奖励、组内方差、KL、梯度、搜索次数、耗时和 GPU；轨迹保存在 outputs/search-grpo/rollout-N.json。GRPO 检查点每步保存，暂不自动清理，请预留磁盘。

## 中断恢复

```bash
python search_train.py sft --config search/pilot.json --resume outputs/search-sft/checkpoint-1
python search_train.py grpo --config search/pilot.json --resume outputs/search-grpo/checkpoint-1
```

选择实际最后一个完整检查点；不要使用 .partial 目录。SFT 使用 Trainer 标准恢复；GRPO 保存适配器、优化器、调度器、Python/Torch/CUDA RNG、步数和信号累计，原子提交目录并检查文件摘要。配置、代码、数据、基座身份必须一致；最大步数也不能在恢复时修改。不会恢复尚未保存的半个采样组，需从最后检查点重新采样。

反馈：`bash scripts/run_search.sh feedback` 导出 dist/cloud-feedback.zip，包含搜索轨迹和日志，不包含模型权重；如以后接私有文档，发送前检查轨迹中的文档内容。

## 当前验收边界

已实现：数据转换入口、检索、动作环境、SFT、GRPO、评测比较、状态保存恢复和遥测代码。
待云端：真实 HotpotQA 下载与转换、tokenizer 长度、1步训练、真实恢复与合并重载、吞吐、有效训练信号和效果。
后续：技术文档语料、可靠的问答与引用标注、成熟并行 rollout 后端。未实现的功能不写成已完成成果。

方法来源：https://arxiv.org/abs/2503.09516
上游：https://github.com/PeterGriffinJin/Search-R1


## 2026-09-18 检查修订

prepare 新增真实 tokenizer 长度预检（search_train.py lengths），覆盖训练动作长度和留出集固定检索初始上下文；动态多轮仍受运行时上限控制。检查点新增 run-identity.json，并强制校验必需文件。旧版 GRPO 检查点缺少新身份文件，不能用新版直接恢复，应保留原代码完成旧运行。

评测现在核对适配器训练身份；比较报告核对同组模型来源及 SFT→合并的权重链。final 补齐 B0 无检索和固定检索测试。旧评测报告缺少 model_provenance 字段，需要在新代码下重新评测，不能混合比较。


## 未合并 SFT 继续 GRPO 与批量计算

使用 `python search_train.py grpo --config search/pilot.json --unmerged-sft --logprob-batch-size 4` 从原 SFT 适配器继续训练。冻结另一份 SFT 适配器作为参考；不使用合并模型，不覆盖原始适配器。允许该明确的跨代码版本 warm start，但仍核对数据、基座、训练配置与依赖，记录来源摘要。

参数 logprob-batch-size 控制同时计算 logprob/反向的动作上下文数，不是采样组大小。逐轨迹 token 归一化保持不变。rollout 仍串行；日志分别记录 rollout_seconds 和 update_seconds，实际吞吐与显存需测量。先用4，OOM时以较小参数启动另一个新实验，不能修改原配置恢复。

本配置输出 outputs/search-grpo-batch4-unmerged；评测命令：`python search_train.py eval --config search/pilot.json --variant B3 --unmerged-sft --logprob-batch-size 4`，结果 reports/search/B3-unmerged-batch4-adaptive-val.json。恢复需同样参数和对应完整检查点。旧比较器默认拒绝跨配置/代码比较，保留旧报告，不手工修改身份。


## G=8 并行轨迹实验

训练命令：`python -u search_train.py grpo --config search/pilot.json --unmerged-sft --group-size 8 --rollout-batch-size 8 --logprob-batch-size 4`。每题8条轨迹，同轮未结束轨迹批量生成；文档搜索仍CPU串行。左填充仅用于批次生成，记录未填充精确前缀与截至EOS的动作；不同剩余上下文预算分批，不因其他长序列额外裁剪短序列。

输出目录 outputs/search-grpo-g8-r8-batch4-unmerged。评测将 grpo 换成 eval 并加 --variant B3，保留全部三个批量参数；输出 reports/search/B3-unmerged-batch4-g8-r8-adaptive-val.json。不可用旧运行检查点恢复新的组大小。保持20步意味着轨迹量从40增到160，效果比较不是等采样预算实验；需同时报告吞吐与总耗时。

若OOM，用 --rollout-batch-size 4 另开实验，G仍为8，分两批完成；输出目录自动隔离。实际GPU性能与显存仍需云端验证。
