# 文档导航

## 当前操作入口

1. [基础数据与云端环境](plans/search-r1-runbook.md)
2. [从原始基座重新训练r32 SFT](plans/2026-09-21-cross-question-fresh-sft.md)
3. [多问题GRPO训练与运行](plans/2026-09-21-cross-question-runbook.md)
4. [固定300题验证集](plans/2026-09-21-validation300.md)
5. [补齐300条规则SFT轨迹](plans/2026-09-21-sft300.md)
6. [资源估算与上下文预算](plans/2026-09-21-cross-question-resource-audit.md)

最新实验摘要以根README为准；操作文档中的路径需对应自己的云端文件。配置由脚本生成，不保证每个文档示例路径在新克隆中已经存在。

## 设计与历史记录

`plans/`里的设计、审计及早期操作文档，和`superpowers/plans/`里的实现计划保留为开发背景。历史参数、计划和当时的实验结论不代表当前已完成状态。不要同时混用不同时期的配置与权重。

## 产物管理

真实数据、模型、检查点、TensorBoard、评测JSON、临时补丁包均保留在本机或云端，不纳入源码仓库。通过生成脚本重建；恢复训练必须匹配原身份文件。
