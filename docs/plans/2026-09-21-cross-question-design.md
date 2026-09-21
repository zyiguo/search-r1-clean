# 更多训练问题、批量评测与跨问题 rollout

## 决策

复用当前单卡 NF4/PEFT 后端，直接实现跨问题动态轨迹队列；本轮不迁移 verl。verl 官方 Agent Loop 支持跨请求协程与 vLLM/SGLang，但需要重做当前 adapter 权重同步、精确动作 token 评分及辅助优势集成。不能把配置文件当作已验证迁移。

来源：https://verl.readthedocs.io/en/latest/advance/agent_loop.html 和 https://verl.readthedocs.io/en/latest/start/agentic_rl.html（2026-09-21 查阅）。

## 实施顺序和验收

1. 动态队列：每条轨迹独立消息/搜索预算，按题与组序号恢复结果顺序；支持 greedy eval 及 none/fixed/adaptive 模式。测试不同问题隔离、不同终止时间、队列补位及串并行等价。
2. 问题覆盖：跨轮随机不放回采样，保存队列顺序/游标/访问次数与 RNG。每次批量采集多个问题组，批内超额有效组记录成本但不更新。逐问题独立优势不变。
3. 数据扩展：从官方 train 增加至2000条（可配置）独立 RL 问题，不以教师轨迹成功作为 RL 准入条件；必须有支持文档，训练题与原 val/test 严格去重。保留原 val/test 字节；扩展 corpus 并记录原数据身份。新语料需重新评测 SFT 基线，不与旧66%直接比较。
4. 集成 grpo_data_dir：原 data_dir 和 SFT adapter 不动；显式新数据 warm start 校验来源数据、原 SFT 配置、基座和依赖。严格恢复只允许同配置/代码/数据。
5. 批量评测：--eval-batch-size 仅影响执行，不改训练配置身份；报告总耗时/吞吐与逐题延迟，避免把并行延迟相加当总耗时；保存批量参数。
6. 全套 CPU 契约测试、命令帮助、打包哈希校验。真实模型批量数值差异、96GB GPU吞吐和显存需云端验证。

用户已确认无需兼容旧 step10 恢复；当前项目直接更新，新训练写入独立输出目录。
