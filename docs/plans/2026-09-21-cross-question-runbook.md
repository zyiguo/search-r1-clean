# 更多问题与跨问题批量训练：AutoDL 操作说明

最新决定：用户从SFT重跑并改为r32/alpha64，请优先使用[从SFT重跑操作说明](2026-09-21-cross-question-fresh-sft.md)。下文保留旧SFT起点的通用流程；不要误用旧r16配置。

代码直接覆盖当前 search-r1-clean 项目；原 SFT adapter 继续作为起点。无需恢复旧 step10。未安装 verl：本版为现有 PyTorch/PEFT 后端的跨问题模型批处理，不是 vLLM 连续批处理或分布式异步训练。检索仍在 CPU 顺序执行。

## 当前80GB配置（按用户要求更新）

- RL问题池2000题；有效 prompt batch目标48，最多尝试512题；不足时按实际有效问题数更新。
- 每题 rollout=5，满额一次更新240条轨迹；每批采集16题，共80条候选轨迹。
- 模型同时生成最多16条活跃轨迹；评分/反向 micro batch=2，跨微批累积梯度后更新一次。prompt batch48不表示48题同时驻留GPU。
- 总上下文上限2048 tokens（包括系统提示、问题、历史动作、检索结果和本轮生成），单动作内部上限512 tokens；最多3次搜索，之后仍有一次回答机会。整条轨迹最多4个动作；每次实际生成上限为 min(512, 2048减去当前前缀长度)，没有滑动窗口或自动丢弃历史；剩余空间不足时标记上下文截断，不保证完成全部3次搜索。
- GRPO学习率1e-6，KL系数0.001；弱奖励证据0.1、动作0.02保持。
- 100个采样轮次，每5轮保存检查点，最后一轮必存（100轮共20份）；验证批量8、greedy。该配置是80GB显存的待测起点，不是已测出的显存最大值。
- 每轮轨迹无损压缩保存为rollout-N.json.gz，diagnose_signal.py和common.read_json均支持读取；指标仍每轮记录。
- epoch洗牌采样并记录问题覆盖率；超过目标的批内候选轨迹计入采样成本但不跨轮复用。

## Qwen3 thinking标签协议

当前固定Qwen3-4B-Instruct-2507。官方说明此版本仅支持non-thinking，无需enable_thinking=False；不要套用其他Qwen3 thinking版本的模板。当前项目是JSON动作协议，未采用原Search-R1的think/search/answer标签协议。

启动时检查本地tokenizer的单轮/检索后多轮模板：不得自动插入think标签、assistant生成前缀必须匹配、教师动作token必须保留生成前缀并包含EOS。异常时在模型加载前报错；生成内容中的思考标签仍按非法动作处理，不静默删除，避免评分与实际采样token不一致。

来源：https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 和 https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py 。CPU测试验证检查逻辑；云端真实tokenizer仍需启动时验证。

## 1. 上传覆盖

上传 dist/grpo-cross-question-patch.zip 和同名 .sha256 到云端项目根目录。压缩包没有外层文件夹。以下环境路径来自此前日志；如已更换环境，请使用实际训练环境。

```bash
cd /root/autodl-tmp/search-r1-clean
source /root/autodl-tmp/crosswoz-stage2/.venv/bin/activate
sha256sum -c grpo-cross-question-patch.zip.sha256
unzip -o grpo-cross-question-patch.zip
python verify_package.py
export POSTTRAIN_ALLOW_CLOUD=1
```

哈希文件使用 LF，无需处理 Windows 回车。更新时应没有训练进程正在运行。补丁不包含训练数据、模型或任何云端配置覆盖。

## 2. 扩展数据

直接使用你已下载的两个 HF distractor train parquet 分片，不需要重新下载；`--train` 也支持恢复后的官方 train JSON 文件：

```bash
python expand_search_data.py \
  --source search/hotpot-pilot \
  --train search/raw-recovery/hotpotqa-hf/distractor \
  --output search/hotpot-grpo-2000 \
  --train-size 2000
```

新目录已存在时会拒绝覆盖。原 val/test 文件逐字节保留，新增训练题按 ID 和规范化问题与全部已有分割去重；无支持文档的题拒绝加入。文档存在不等于当前 top-k 一定能检索到，也不等于语义充分。新增上下文会扩大共享语料，不能把原0.66直接当作新语料下的基线。

## 3. 派生新配置

必须从云端实际使用的3epoch SFT配置派生，不重新猜测 SFT 步数：

```bash
python prepare_grpo_experiment.py \
  --source search/pilot-sft-3epoch-b16.json \
  --output search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json \
  --steps 100 --profile rtx6000-80gb --checkpoint-every 5 \
  --grpo-data-dir search/hotpot-grpo-2000
python search_train.py validate --config search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json
```

生成器直接写入 unmerged SFT 起点、prompt batch48、G=5、rollout batch=16、logprob batch=2，无需再传旧的覆盖参数。输出目录自动添加新后缀；生成器打印其准确路径。profile优先于groups/attempts/question-batch-size参数；它也记录原SFT配置与GRPO运行时改动，允许复用原adapter，不修改SFT训练历史。

## 4. 批量评测 SFT，然后训练

```bash
python -u search_train.py eval \
  --config search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json \
  --variant B1 --eval-batch-size 8 \
  --output reports/search/B1-p48-g5-ctx2048-expanded-val.json
python -u search_train.py grpo \
  --config search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json
```

不追加 --unmerged-sft 等旧覆盖参数，否则会再改变输出目录后缀。GPU真实性能未在本地验证；优先查看 TensorBoard 的采样耗时、更新耗时、轨迹吞吐、有效问题比例、问题覆盖与显存，而不是只看 loss。

## 5. 批量评测 GRPO

```bash
python -u search_train.py eval \
  --config search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json \
  --variant B3 --eval-batch-size 8 \
  --output reports/search/B3-p48-g5-ctx2048-expanded-val.json
python search_compare.py \
  reports/search/B1-p48-g5-ctx2048-expanded-val.json \
  reports/search/B3-p48-g5-ctx2048-expanded-val.json
```

中间检查点：`python scripts/eval_grpo_checkpoint.py --checkpoint 实际新输出目录/checkpoint-10 --eval-batch-size 8 --output reports/search/B3-crossq-step10-val.json`。

执行参数 eval_batch_size 独立记录在报告 execution 中；wall_seconds 是评测生成/检索墙钟时间，不含模型加载；questions_per_second 是吞吐；amortized_seconds_per_question 是平均摊销耗时。每题 seconds 是从入队到完成的延迟，彼此重叠，不能求和当总耗时。GPU批处理可能产生数值差异，严格复现实验应保持相同评测批量。

## 验证边界

CPU测试覆盖跨题上下文隔离、动态补位、逐题优势、零信号/超额组、完整epoch覆盖、采样器状态恢复、greedy批量评测、数据去重与保留验证集、SFT来源校验，以及实际训练控制流的微型可微模型测试。没有在本地启动真实模型训练，没有测得GPU加速倍率或新EM。


## 资源估算

本轮修改后的计算与风险边界见同目录 `2026-09-21-cross-question-resource-audit.md`。CPU纯计算，不加载模型：

```bash
python estimate_search_resources.py \
  --config search/pilot-sft3epoch-grpo100-p48-g5-ctx2048.json \
  --output reports/search/resource-estimate.json
```

它优先读取本地模型的config.json；模型未下载时使用官方Qwen3-4B-Instruct-2507维度并注明来源。已有云端配置必须重新派生；覆盖代码不会自动改写旧配置。
