# NextFire 对照实验

日期：2026-09-19。实验使用本地已有人声 stem，不修改源样本；结果目录在 `temp/nextfire-20260919/`，其中 `review.html` 可逐句播放，两个模型的字符高亮会同步显示。

```bash
conda run -n ktv python -u tools/compare_nextfire.py \
  --output temp/nextfire-20260919 \
  --japanese-model models/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese/snapshots/cf031e020336460d15a417eba710bbc5bb43be9a \
  --nextfire-model ../ref/mms-300m-ForcedAligner-karaoke-ja-Latn
```

## 样本

播放对照页时使用支持 HTTP Range 的服务器（ktv 环境已安装 Flask）：

```bash
conda run --no-capture-output -n ktv python tools/serve_review.py \
  --directory temp/nextfire-full-3346334398 --host 0.0.0.0 --port 18081
```

访问 `/review.html`。普通 `python -m http.server` 不支持音频分段请求，
可能导致点击句子后播放位置回到开头。播放器会在音频元数据加载完成后
应用最近一次点击的位置，切换原曲／人声也会保留待跳转的位置。

- `557579321_ヨルシカ_ただ君に晴れ`：8 个纯日语行；当前模型和 NextFire 都是 8/8 通过质量门。
- `3346334398_HALCALI_おつかれSUMMER`：8 个日英混写行；当前模型 3/8，NextFire 6/8 通过。NextFire 的拉丁目标覆盖率为 0.98–1.0，但部分显示字符仍需要时间投影修正。
- `1851578144_東京事変_孔雀`：日语和英语段落；当前模型 4/8，NextFire 3/8。英语长句的 NextFire emission 分数仍低，说明罗马化本身没有解决所有英语歌唱问题。

这些“通过”数字是当前质量门（CTC 平均分、词表覆盖和正时长）结果，**不是人工边界准确率**。没有人工 TextGrid，不能据此断言 NextFire 在时间误差上更好。实验实际说明了两点：

1. NextFire 的拉丁目标路线已经能在现有日语样本上运行，且对一组真实日英混写素材减少了 fallback；
2. 英语仍需要语言感知的发音候选或专门的英语歌声数据，不能只把英文表面拼写塞进日语罗马音流程。

实验还暴露了显示层问题：一个英文词或日语表记可能对应多个声学 mora，显示字符需要局部切分和单调性修正。该修正已纳入公共 artifact 的 `display_units`，不会改变默认 `japanese` profile。
