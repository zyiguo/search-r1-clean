# 项目缺漏检查（2026-09-18）

酒店/CrossWOZ 链路已删除；本审计中与酒店入口相关的条目不再适用。搜索链身份校验、长度预检与包校验规则仍然有效。

## 已确认并修复（搜索链）
- P1：缺少真实 tokenizer 长度预检。新增 lengths 入口并接入 prepare，训练前检查动作生成预算；长度失败不先创建运行目录。
- P1：GRPO 检查点摘要清单不要求必需文件，也未绑定运行身份。强制完整清单、run-identity、目录范围和恢复步数检查。
- P1：评测适配器只验证 task，可能混入另一轮训练。核对完整训练 config/code/data/model 身份。
- P1：比较报告不校验基础/合并模型来源和 SFT 合并链。新增模型来源链和 EM 汇总一致性校验。
- P2：final 缺少无检索与固定检索对照。新增两组 B0 测试。

## 未完成的实验与产品工作
- 真实 tokenizer/GPU/训练/合并重载/恢复以云端实测为准；CPU 测试不能替代。
- HotpotQA 正式文件不随包分发，真实转换与教师轨迹人工抽查尚需完成。
- 教师搜索用 gold supporting titles，属于 oracle-assisted 冷启动；文档命中不保证800字符截断后仍有完整证据。
- 共享小语料 BM25 不是 fullwiki；仅严格问题去重，不保证语义去重或无预训练污染。
- GRPO 在强 SFT 上是否稳定超过 SFT val EM，仍待 `pilot-sft-metrics` 之后的实验。
- 检查点每步保存，长跑需关注磁盘。
- 当前评测会在完成全部题目后写报告，中途失败需重跑；尚未实现逐题恢复。

## 兼容边界
删除酒店链后，旧 `train_qlora.py` / `configs/*` / CrossWOZ 数据入口不再存在。搜索实验请使用 `search/*.json` 与 `search_train.py`。本地验证结果见 `reports/local-verification.json`；压缩包由 `build_package.py` 重新生成。
