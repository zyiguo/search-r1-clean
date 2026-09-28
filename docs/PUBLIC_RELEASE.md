# 发布云端实验产物

本项目的 Git 仓库只保存代码和说明。完成 GRPO 检查点评测后，把本轮 GRPO 评测报告、处理后的 SFT/GRPO 数据集、SFT/GRPO LoRA adapter 作为同一 GitHub Release 的附件发布。SFT 评测报告不在本轮发布范围内。不要上传运行缓存、优化器状态、原始下载文件或 Qwen 基座权重。

发布前，确认 GRPO 报告对应本次 20 步检查点。脚本核对问题 ID、记录的训练身份和检查点路径，但不重算文件哈希。Release 不作 SFT 与 GRPO 的性能对比。

在 AutoDL 终端运行以下命令；把 `GRPO_REPORT` 改成刚生成的 20 步评测报告路径：

```bash
cd /root/autodl-tmp/search-r1-clean
source /root/autodl-tmp/crosswoz-stage2/.venv/bin/activate
mkdir -p scripts
curl -fL --retry 3 https://raw.githubusercontent.com/zyiguo/search-r1-clean/main/scripts/prepare_public_release.py -o scripts/prepare_public_release.py

GRPO_REPORT=reports/search/B3-sft300-step20-20260928-144513.json
RELEASE_DIR=dist/release-sft300-grpo20-only-grpo-report

python scripts/prepare_public_release.py \
  --grpo-report "$GRPO_REPORT" \
  --output-dir "$RELEASE_DIR"

cat "$RELEASE_DIR/RELEASE-NOTES.md"
ls -lh "$RELEASE_DIR"/*.zip
```

如果报告中的路径不是当前云端项目路径，可给打包脚本补充 `--sft-data`、`--grpo-data`、`--sft-output`、`--grpo-checkpoint`。发布前自行确认压缩包里的内容，因为此流程按要求不重算文件哈希。

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
