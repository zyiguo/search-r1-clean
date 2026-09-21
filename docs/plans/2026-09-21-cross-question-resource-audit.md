# 2048总上下文：逻辑审计与资源估算

## 当前配置

总上下文2048 tokens；单动作内部安全上限512，实际输出预算min(512,2048-prefix长度)。最多3次搜索+1次回答机会。prompt batch指目标48个有信号的问题组，G=5，目标更新240条轨迹。每批采16题、生成并发16、反向微批量2。最大尝试512题/轮。学习率1e-6，KL0.001。NF4基座共享，FP32 LoRA r16，七类线性层，100轮，每5轮及末轮保存检查点；保留所有计划内检查点，不删除旧实验。

## 已修正

1. 将误设的context4096/action2048纠正为context2048/action512。测试覆盖单例与批量：前缀2000只能生成48 tokens、2020只能28 tokens、已超限前缀不进入模型。
2. 增加context_exhausted/context_limit/action_limit区分及每轮TensorBoard计数。达到最大搜索次数后仍允许回答；继续搜索才是search_limit。
3. 扩展数据的诊断默认选择grpo_data_dir；原实现仍选旧data_dir，可能把新增题/文档当缺失。日志按轮次数字顺序读取。
4. 新profile每轮完整轨迹用流式JSON+gzip无损保存，避免未压缩的token数组/多轮历史造成磁盘与JSON字符串内存峰值；旧JSON日志仍可读取。
5. 检查点按5轮保存，末轮非5倍数也保留；TensorBoard/轨迹仍每轮。SFT来源校验允许显式记录的运行时参数变化，仍拒绝SFT学习率等来源参数篡改。

## 核查后保留的边界

- 检索结果当前每篇800字符、top-k2。字符不是token；三轮累计最多4800字符，另有JSON包装、提示、历史动作，所以2048上下文可能在第三次搜索后不足。不会静默截断证据或丢弃历史；截断率需要云端观察。最多3次不保证每题3次。
- 48是有效组目标，不是总采样量上限。若有效组率40%，约需120题×5=600条候选轨迹；最多512×5=2560条/轮。无信号的轮次会跳过优化器，所以100轮不等于100次更新。
- 每题内部归一化优势；观察文本无loss；奖励中的证据召回只是支持文档ID召回，不保证模型可见部分含答案。弱奖励不能代替EM改进验证。
- 当前EM规范化是NFKC/大小写/空白；它不是完整官方HotpotQA EM（未采用官方标点/冠词处理）。本次不改变奖励定义，否则旧指标和训练目标同时变化。
- 当前是同GPU上交替生成/反向，不是verl异步推理；长短prefix分桶后实际生成batch可能小于16。不会为了拼批量擅自缩短单题预算。
- 原SFT与GRPO必须在相同新语料、搜索预算、总上下文、验证batch下重新评测。新结果不能直接和旧0.66当作同协议比较。

## 可计算的资源项（GiB，1GiB=2^30 bytes）

模型维度来自官方config：36层、hidden2560、intermediate9728、32个Q头、8个KV头、head_dim128、词表151936，输入输出embedding共享。

来源：https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json

按矩阵维度计算模型参数4,022,468,096；LoRA参数33,030,144。

| 项目 | 计算/假设 | 大小 |
|---|---|---:|
| 原始BF16模型文件 | 4,022,468,096×2 | 7.49 GiB |
| NF4基座常驻估计 | 线性层约0.52字节/参数，embedding/norm FP32 | 3.21 GiB |
| policy+reference LoRA、policy梯度、Adam双状态 | 33,030,144×4×5 | 0.615 GiB |
| 满长度16并发KV缓存 | 2×36×8×128×16×2048×(2或4字节) | 4.5–9.0 GiB |
| 反向微批量2的完整FP32 logits | 2×2048×151936×4 | 2.32 GiB |
| 单份动作FP32 softmax | 2×512×151936×4 | 0.58 GiB |
| 各层重计算输入量级 | 36×2×2048×2560×4 | 1.41 GiB |
| 单个FP32 adapter | 33,030,144×4 | 0.123 GiB |
| 单个checkpoint的adapter+Adam | 33,030,144×12 | 0.369 GiB |
| 20份完整checkpoint，加tokenizer/元数据预算 | 20×0.369+20×16MiB | 约7.70 GiB |

PEFT的prepare_model_for_kbit_training会将非Params4bit的半精度参数转FP32，不能把所有内存都按BF16计算。policy/reference共享同一份基座；参考策略不是另一个完整4B模型。生成KV与训练反向并非同时达到峰值，上表不能简单全部相加。当前Transformers生成路径会在支持时设置logits_to_keep=1，而训练评分仍生成完整logits。

实现依据：https://raw.githubusercontent.com/huggingface/peft/v0.17.1/src/peft/utils/other.py
以及 https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py

## 容量规划范围（不是实测或严格上界）

| 资源 | 本次规划 |
|---|---|
| GPU显存 | 估计约20–40 GiB峰值量级，建议留到48 GiB以上可用空间；80GB卡预计有余量，取决于实际KV/attention dtype、SDPA后端和分配器 |
| 主机RAM | 估计8–16 GiB工作集量级，建议32 GiB容器配额；全量parquet转Python对象、模型加载、BM25和上轮轨迹引用会影响峰值 |
| 从头部署磁盘 | 预留40–50 GiB，包含约7.5 GiB模型、7.7 GiB新检查点、环境/数据/压缩轨迹和临时写入余量；不包含旧实验 |
| 已有模型/环境的本轮额外磁盘 | 建议至少25–30 GiB空闲；检查点7.7 GiB较确定，其余取决于轨迹长度、有效组率、压缩率和数据扩展 |

RAM的可算小项：最坏512题×5条×4动作×2048个Python token整数，以36字节/int+list slot估算约0.703GiB；不含消息字符串、dict、拒绝的超长前缀或暂时仍有引用的上轮轨迹。240条选中轨迹old/ref标量logprob上限约0.0037GiB，绝不是保存整个词表logits到CPU。

压缩轨迹暂按数GiB预留，但不存在与文本内容无关的精确大小保证；最坏长期低有效率会持续触发2560候选/轮。跑到checkpoint-5后用实际文件增量乘20，可显著收紧磁盘估算。GPU使用现有gpu/*遥测和train/peak_vram_bytes核对；nvidia-smi使用量包括CUDA保留/工作区，通常高于torch已分配量。

完整算术输出：reports/search/resource-estimate-ctx2048.json。estimate_search_resources.py可在云端用实际模型config重复计算。本地未启动真实模型、未连接云端测量，因此上述范围不能作为已验证的容量保证。
