# 多问题组 GRPO：云端运行与指标

## 行为

默认旧配置仍是一题一组、不滤零方差组。新配置显式设置：

```json
{
  "grpo_groups_per_update": 4,
  "grpo_max_group_attempts": 16,
  "grpo_filter_zero_variance": true,
  "grpo_evidence_weight": 0.1,
  "grpo_protocol_weight": 0.02
}
```

每个采样轮在训练问题中不放回抽样；问题数少于 16 时最多遍历训练集一次。每题独立生成 G 条轨迹、分别计算 EM、证据覆盖和协议分量并在题内标准化，然后按 1、0.1、0.02 合成优势（不再归一化）。收集到 4 个合成优势非零的问题组立即停止；达到预算或问题耗尽时，使用实际有效组更新。没有有效组则不调用优化器/调度器，但保存该轮检查点。过滤组既不做任务损失，也不做 KL。

每条有效轨迹的损失先除以该轨迹动作 token 数，再除以实际有效轨迹总数。多组不会直接放大梯度尺度。问题组之间串行采样，同题轨迹可批量生成；logprob-batch-size 仍控制评分/反向批次，不需要一次把全部组放入 GPU。

新模式的 grpo_steps=100 表示 100 个有界采样轮，实际优化器更新数可能少于 100。G=8、最大尝试16组时最多采样128条轨迹/轮、12800条/100轮，不能与旧版100组/800条轨迹称作等预算比较。过滤使训练偏向当前有时能答对的问题；不保证验证 EM 提升。

## 使用既有 3 epochs SFT

将更新代码放到云端前，保存旧版代码以便恢复/评测旧实验；本次更改会改变代码身份，不能恢复旧 GRPO 检查点。SFT adapter 支持受校验的跨版本 warm start；不修改其 identity.json。训练启动后到评测完成前不要再更新根目录 Python 文件。

```bash
cd /root/autodl-tmp/search-r1-clean
source .venv/bin/activate
export POSTTRAIN_DATA_ROOT=/root/autodl-tmp
source scripts/env_6000d.sh

python prepare_grpo_experiment.py \
  --source search/pilot-sft-3epoch-b16.json \
  --output search/pilot-sft3epoch-grpo100-multi-aux.json

python -u search_train.py grpo \
  --config search/pilot-sft3epoch-grpo100-multi-aux.json \
  --unmerged-sft --group-size 8 --rollout-batch-size 4 --logprob-batch-size 4

python -u search_train.py eval \
  --config search/pilot-sft3epoch-grpo100-multi-aux.json \
  --variant B3 --unmerged-sft --group-size 8 --rollout-batch-size 4 --logprob-batch-size 4 \
  --output "reports/search/B3-sft3epoch-grpo100-multi-aux-val-$(date +%Y%m%d-%H%M%S).json"
```

生成器保留真实的 SFT 步数、验证频率和所有 SFT 参数，只增加 GRPO 字段；目标配置存在时拒绝覆盖。若源 grpo_output 为 outputs/search-grpo-3epoch-b16，最终输出是 outputs/search-grpo-3epoch-b16-multi4-a16-n100-aux-g8-r4-batch4-unmerged。

恢复本次新实验时，保留所有训练参数，加 --resume 指向此目录的完整 checkpoint-N。N 是采样轮数而非实际更新次数；随机状态及累计采样统计随检查点恢复。

## 重点观察

- sampled_groups / sampled_trajectories / sampled_generated_tokens：包含被过滤组的真实采样成本。
- effective_group_fraction：尝试组中合成优势非零的比例，不是仅对保留组计算。
- selected_groups / collection_target_met：本轮真正用于更新的组数及是否凑齐目标。
- optimizer_update：本轮是否更新；signal_updates：累计有合成优势信号且梯度非零的更新次数（可能仅来自辅助奖励）。
- reward：所有采样轨迹的平均奖励；reward_std：各题组内标准差的均值，不是跨题混算。
- rollout_seconds / update_seconds：采样与更新耗时，不包括之后的检查点磁盘写入。
- result.json 的 sampling_totals：跨恢复累计的采样成本、实际更新数和跳过轮数。

rollout-N.json 的 groups 保留所有组的 sample_id、奖励、优势、selected 和轨迹；diagnose_signal.py 兼容此格式及旧单组格式。采样率上升或信号门槛通过都不等于 EM 提升，仍需固定验证集和相同 SFT 来源比较。

CPU 测试可验证控制流和损失归一化；真实 CUDA 性能、显存、保存重载和效果需云端验证。


## 辅助奖励（已直接启用在新配置生成器）

- EM：原始最终答案精确匹配，原奖励和验证评分不变。
- evidence：累计命中的不同支持文档数 / 支持文档数；无支持标签时为0，重复文档只计一次。它等于新增覆盖率增量之和，但当前实现仍是轨迹级奖励，不做逐动作 return-to-go；文档命中不保证截断文本包含答案。
- protocol：invalid_action 或 search_limit 为 -1，其余为0。合法动作不按次数给正奖励；truncated 是预算/生成截断，不在此当作非法动作惩罚。
- 每个分量分别在同题内标准化，最终优势为 A_EM + 0.1*A_evidence + 0.02*A_protocol，不再次标准化。过滤以实际合成优势非零为准，EM全对/全错不再必然被过滤。
- 日志保留 rewards 和 reward 为纯 EM；reward_std 仍是 EM 组内标准差均值。不要再用 reward_std>0 判断全部训练信号。
- 新增 evidence_reward、protocol_reward、em_effective_groups、evidence_effective_groups、protocol_effective_groups、auxiliary_only_groups。分量有效组计数可重叠；effective_groups 是合成优势真正非零的组数。
- 支持标签只在训练评分中读取，生成器仍只接收问题和检索历史。评测仍只报告原 EM，不用辅助奖励提升评测分数。
- 新配置默认输出以 -aux 隔离已有训练结果；仍在原项目内运行。配置/代码改变后不可恢复旧检查点；从同一 SFT adapter 开始新实验。
