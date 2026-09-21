# Search-R1 风格 Qwen3-4B 适配计划

## 定位
用户批准保留 Qwen/Qwen3-4B-Instruct-2507 并迁移到搜索任务。实现 SFT → 多步搜索 GRPO → 独立评测，属于方法适配，不是论文原设置复现。仓库仅保留搜索入口；酒店历史链路已删除。复用基础模型下载、NF4 加载、LoRA 和 TensorBoard。

## 方案选择
选用现有 PyTorch/Transformers/PEFT 的单卡串行采样后端，实现一次新鲜采样对应一次 GRPO 更新。避免盲目安装上游旧 veRL/vLLM；未来有吞吐测量后再替换成熟并行后端。工具协议采用严格 JSON search/answer 动作，无强制思考标签，适配非思考 Instruct 模型。奖励只有最终答案 exact match；不为搜索次数或格式单独加分。

## 交付顺序
1. 独立配置、语料/QA/示范轨迹契约、来源清单与划分泄漏检查。
2. 可复现 CPU BM25 检索、有限步数的交互环境、严格动作解析。
3. SFT 每个动作只训练 assistant token；工具返回放入 observation，屏蔽损失。
4. GRPO 对同题 G 条轨迹求组相对优势；记录精确采样 token，在同一上下文计算 policy/ref logprob，仅 assistant token 参与损失。
5. 单卡 bf16、一次更新一次采样、截断失败、零方差记录、KL、新适配器参考模型、完整检查点恢复。
6. 无检索/固定检索/自适应检索评测；输出答案、动作、命中文档与成本；CPU 单测和包校验。

## 边界
不在本地下载模型或训练。内置微型虚构语料仅测试环境，不能作为正式数据或效果证据。正式 corpus/train/val/test 必须显式提供来源；不从测试 gold 构造检索结果。训练只能读取 train/val，test 需显式 final-test。完整语料内存版 BM25 面向受控小语料，不能直接载入完整 Wikipedia；大语料后续接独立索引服务。

云端依次验证 tokenizer、一步 SFT、合并、一步 RL、保存重载、恢复，测量吞吐后扩大训练。单卡可行性是工程假设，未实测不承诺耗时/显存或效果提升。

来源：https://arxiv.org/abs/2503.09516
上游：https://github.com/PeterGriffinJin/Search-R1
