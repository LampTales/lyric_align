# lyric_align 实验目录

`samples/` 是从原项目 `local/outputs/` 复制的长期回归样本，不会随原仓库变化。

样本目录仅用于本机长期回归，整个 `samples/` 已加入 `.gitignore`，不会进入 Git 提交。若要在另一台机器复现实验，需要另外准备这些样本文件。

当前实验环境为 conda 的 `lyric`（Python 3.11），依赖清单见 [environment.yml](/Users/lamptales/remote/cloudmusic2ktv/lyric_align/environment.yml)。已经安装 `pykakasi`、SudachiPy/UniDic 和 `pyopenjtalk`；基线会优先使用 pykakasi，后续实验可直接比较其他后端。

模型权重和声学中间文件统一放在当前目录的 `models/` 或 `artifacts/`，这两个目录已加入 `.gitignore`，不会进入 Git 提交。

Demucs 使用本地模型缓存时设置：

```bash
export HF_HOME="$PWD/models/huggingface"
```

## 运行无依赖基线

```bash
conda activate lyric
python align_baseline.py
python -m http.server 8765 --bind 0.0.0.0 --directory .
```

浏览器打开 <http://localhost:8765/results/>。如果服务器在远程 MacBook 上，将 `8765` 端口通过 SSH 隧道转发到本机，例如：

在 ZeroTier 网络中，也可以直接从同网电脑打开：

```text
http://10.10.10.20:8765/results/
```

服务器需要监听 `0.0.0.0`：

```bash
python -m http.server 8765 --bind 0.0.0.0 --directory .
```

如果不使用 ZeroTier，才需要通过 SSH 隧道转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

页面可以选择歌曲、播放音频，并观察当前句和 mora 的高亮。当前算法只是句级时间戳内等速插值；没有安装 G2P 时，汉字会显示为 `unresolved`，这是有意保留的失败标记。

## 可选日语 G2P

基线会自动探测 `pykakasi` 或 `pyopenjtalk`，但不会自动下载模型。安装其中一个后重新运行 `align_baseline.py` 即可比较文本侧结果：

```bash
python -m pip install pykakasi
```

后续再单独加入 Sudachi/UniDic、Demucs 和 CTC 强制对齐，避免把大模型依赖和数据格式验证混在第一步。

## 比较日语 G2P 后端

```bash
conda activate lyric
python compare_g2p.py
```

报告地址为 <http://10.10.10.20:8765/results/g2p/>。绿色代表三套读音一致，黄色代表两套一致，红色代表三套都不同；分歧报告用于挑选需要人工验证的歌词，并不自动把多数结果当成真值。

候选读音决策和对齐输入可用：

```bash
python prepare_reading.py
```

报告地址为 <http://10.10.10.20:8765/results/reading/>。三方冲突行默认采用 Sudachi 结果，但标记为 `low/manual_review`，不会被视为最终真值。

生产式流程不要求运行三个后端，可以显式选择一个：

```bash
python prepare_reading.py --backend openjtalk
# 或 --backend sudachi / --backend pykakasi
```

单后端结果标记为 `single_backend`，会自动继续生成，不会阻塞手机端使用；`consensus` 仅用于离线比较和发现潜在歧义。

## 句内活动窗口基线

```bash
python activity_baseline.py
```

结果在 `results/activity/`。它使用 FFmpeg + NumPy 的短时能量，只修剪句级锚点内明显的低能量边界；伴奏也会产生能量，因此这不是人声检测，所有调整都带有 warning，供后续人声分离/CTC 对比。

可视化对比页面：<http://10.10.10.20:8765/results/activity/>。灰色为原句级窗口，绿色为能量分析后窗口。

Demucs 分离后，可以比较原混音与人声 stem 的活动窗口：

```bash
python compare_vocal_activity.py samples/1851578144_東京事変_孔雀 \
  artifacts/demucs_local/htdemucs/audio/vocals.wav
```

页面：<http://10.10.10.20:8765/results/vocal_activity/>。

## CTC 对齐探针

Demucs 人声 stem 准备好后，可以对一首歌的前几句运行 CTC 探针：

```bash
export HF_HOME="$PWD/models/huggingface"
python ctc_align.py --max-lines 5
```

结果写入 `results/ctc_probe.json`。该脚本只用于判断日语 wav2vec2 在歌声上的字符后验是否有用，尚未接入视频渲染，也不会自动替换现有时间轴。

本次 `孔雀` 前五句的浏览器检查页：<http://10.10.10.20:8765/results/ctc_probe.html>。

简单歌曲 `羊文学／なつのせいです` 的前八句：

```bash
python ctc_align.py --song 'samples/1867888452_羊文学_なつのせいです' \
  --vocals artifacts/demucs_natsu/htdemucs/audio/vocals.wav \
  --reading 'results/reading_natsu/1867888452_羊文学_なつのせいです.json' \
  --max-lines 8 --out results/ctc_natsu.json
python ctc_mora.py --input results/ctc_natsu.json --out results/ctc_natsu_mora.json
```

检查页：<http://10.10.10.20:8765/results/ctc_natsu.html>。这八句均通过当前质量门控；这只说明模型在简单样本上能形成连续字符路径，仍需与人工/更可靠标注对照。

完整 36 行的探针结果位于 `results/ctc_natsu_all.json`，浏览器检查页为 <http://10.10.10.20:8765/results/ctc_natsu_all.html>；mora 聚合结果为 `results/ctc_natsu_all_mora.json`。

将 CTC 字符时间聚合到 mora，并启用自动质量门控/回退：

```bash
python ctc_mora.py
```

`results/ctc_mora.json` 中 `alignment_status=ctc` 表示通过质量门控；低分、缺失字符或路径异常会自动使用句级插值，并记录 warning，不需要终端用户人工修正。
