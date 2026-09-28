# 发布云端实验产物

本项目的 Git 仓库只保存代码和说明。完成 GRPO 检查点评测后，把完整评测报告、处理后的 SFT/GRPO 数据集、SFT/GRPO LoRA adapter 作为同一 GitHub Release 的附件发布。不要上传运行缓存、优化器状态、原始下载文件或 Qwen 基座权重。

发布前，确认两份报告对应本次 SFT 和 GRPO 20 步检查点。脚本核对问题 ID、记录的训练身份和检查点路径，但不重算文件哈希；若 SFT 与 GRPO 评测使用不同检索语料，Release 说明会标记这两个 EM 不是仅模型改变的受控对比。

在 AutoDL 终端运行以下命令；把 `GRPO_REPORT` 改成刚生成的 20 步评测报告路径：

```bash
cd /root/autodl-tmp/search-r1-clean
source /root/autodl-tmp/crosswoz-stage2/.venv/bin/activate
git pull --ff-only origin main

SFT_REPORT=reports/B1-sft-t300-val300-20260928-095940.json
GRPO_REPORT=reports/search/请替换为20步评测报告.json
RELEASE_DIR=dist/release-sft300-grpo20-20260928-nohash

python scripts/prepare_public_release.py \
  --sft-report "$SFT_REPORT" \
  --grpo-report "$GRPO_REPORT" \
  --output-dir "$RELEASE_DIR"

cat "$RELEASE_DIR/RELEASE-NOTES.md"
ls -lh "$RELEASE_DIR"/*.zip
```

如果两份报告中的路径不是当前云端项目路径，可给打包脚本补充 `--sft-data`、`--grpo-data`、`--sft-output`、`--grpo-checkpoint`。发布前自行确认压缩包里的内容，因为此流程按要求不重算文件哈希。

确认 Release 说明和附件内容适合公开后，在云端登录自己的 GitHub 账号并发布：

```bash
gh auth status
# 若未登录：gh auth login

gh release create exp-sft300-grpo20-20260928 \
  --repo zyiguo/search-r1-clean \
  --target main \
  --title 'SFT300 + GRPO step 20 artifacts' \
  --notes-file "$RELEASE_DIR/RELEASE-NOTES.md" \
  "$RELEASE_DIR"/*.zip

gh release view exp-sft300-grpo20-20260928 \
  --repo zyiguo/search-r1-clean --json url,assets
```

每个附件必须小于 GitHub Release 的 2 GiB 限制，脚本会在超限时停止。LoRA adapter 可以在固定版本的 Qwen 基座模型上加载；这里不发布基座权重。数据来自 HotpotQA，按 CC BY-SA 4.0 标注来源和加工说明。公开前检查报告中的逐题轨迹及数据内容，避免发布不想公开的实验信息。
