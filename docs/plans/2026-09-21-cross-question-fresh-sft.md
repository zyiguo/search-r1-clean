# 从 SFT 重跑：r32 / alpha64 / 3 epoch

本轮用户明确选择r32、alpha64。SFT从原始Qwen3-4B-Instruct-2507基座重新训练，不加载r16 adapter；GRPO从新r32 SFT adapter启动。总上下文2048，单动作内部上限512，最多搜索3次。SFT继续使用原teacher数据search/hotpot-pilot；GRPO扩展2000题不含完整teacher actions，不能直接喂入SFT。

SFT学习率沿用源SFT配置（之前1e-4），micro batch16、梯度累积1、开启梯度检查点。按实际教师动作数、按问题划出的训练/验证分割计算3 epoch步数，不硬编码184或400步。每5步SFT验证/保存，保留至多2份（含best）；最终adapter由held-out action loss选best，可能来自第3epoch之前。GRPO100轮，每5轮及最后一轮保存，prompt48、G5、生成并发16、反向微批2、lr1e-6、KL0.001。

以下命令只在AutoDL执行；本地未训练或上传。先把最新覆盖包与.sha256上传到现有项目根目录。

## 1. 进入项目、激活现有环境、覆盖代码

```bash
cd /root/autodl-tmp/search-r1-clean
source /root/autodl-tmp/crosswoz-stage2/.venv/bin/activate
export POSTTRAIN_ALLOW_CLOUD=1
export PYTHONUNBUFFERED=1
sha256sum -c grpo-cross-question-patch.zip.sha256
unzip -o grpo-cross-question-patch.zip
python verify_package.py
```

## 2. 从实际云端旧SFT配置派生新r32配置

```bash
python prepare_sft_experiment.py \
  --source search/pilot-sft-3epoch-b16.json \
  --output search/sft-r32-e3-b16-ctx2048.json \
  --data-dir search/hotpot-pilot \
  --rank 32 --epochs 3 --micro-batch 16
python search_train.py validate --config search/sft-r32-e3-b16-ctx2048.json
python search_train.py lengths --config search/sft-r32-e3-b16-ctx2048.json
python -u search_train.py sft --config search/sft-r32-e3-b16-ctx2048.json
```

SFT新输出固定为outputs/search-sft-r32-e3-b16-ctx2048。生成器拒绝覆盖已有配置或非空新输出目录，避免意外复用adapter。它会打印训练/验证题数、动作数、每epoch步数和总步数。环境路径与旧SFT配置文件名如已变化，应替换为实际路径。

## 3. 准备或复用已扩展的RL数据

如果search/hotpot-grpo-2000/manifest.json已经存在，可跳过扩展，后续validate会校验哈希。

```bash
if [ ! -f search/hotpot-grpo-2000/manifest.json ]; then
  python expand_search_data.py \
    --source search/hotpot-pilot \
    --train search/raw-recovery/hotpotqa-hf/distractor \
    --output search/hotpot-grpo-2000 --train-size 2000
fi
python prepare_grpo_experiment.py \
  --source search/sft-r32-e3-b16-ctx2048.json \
  --output search/grpo-r32-e3-p48-g5-ctx2048.json \
  --steps 100 --profile rtx6000-80gb --checkpoint-every 5 \
  --grpo-data-dir search/hotpot-grpo-2000
python search_train.py validate --config search/grpo-r32-e3-p48-g5-ctx2048.json
```

## 4. 相同RL检索环境下评测新SFT，然后GRPO

```bash
python -u search_train.py eval \
  --config search/grpo-r32-e3-p48-g5-ctx2048.json \
  --variant B1 --eval-batch-size 8 \
  --output reports/search/B1-r32-e3-expanded-val.json
python -u search_train.py grpo --config search/grpo-r32-e3-p48-g5-ctx2048.json
python -u search_train.py eval \
  --config search/grpo-r32-e3-p48-g5-ctx2048.json \
  --variant B3 --eval-batch-size 8 \
  --output reports/search/B3-r32-e3-expanded-val.json
python search_compare.py reports/search/B1-r32-e3-expanded-val.json reports/search/B3-r32-e3-expanded-val.json
```

无需--unmerged-sft或旧的G8覆盖参数，生成器已写入所需设置。不要运行旧run_search.sh的自动merge流程，本轮直接用新SFT adapter。

## r32资源变化

LoRA参数由33,030,144增至66,060,288。单adapter约0.246GiB；单GRPO完整checkpoint约0.738GiB（adapter+Adam），20份加tokenizer预算约15.08GiB，比r16多约7.38GiB。policy/reference/梯度/Adam常驻LoRA部分约1.23GiB；KV缓存与上下文logits不因r翻倍，因此总显存并不翻倍。

SFT是另一个阶段：micro16且有监督loss，其峰值不能直接套用GRPO micro2的20–40GiB估算；已启用梯度检查点，但80GB真实峰值仍需首轮遥测验证。若需降低SFT微批量，应重新生成配置和epoch步数，不直接修改旧配置后续跑。

```bash
python estimate_search_resources.py \
  --config search/grpo-r32-e3-p48-g5-ctx2048.json \
  --output reports/search/resource-estimate-r32.json
```

该脚本估算GRPO而非SFT显存；从头部署建议比r16额外预留约10GiB磁盘（新GRPO检查点增量及SFT检查点）。旧r16实验文件不自动删除。
