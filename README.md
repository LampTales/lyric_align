# lyric_align 实验目录

`samples/` 是从原项目 `local/outputs/` 复制的长期回归样本，不会随原仓库变化。

样本目录仅用于本机长期回归，整个 `samples/` 已加入 `.gitignore`，不会进入 Git 提交。若要在另一台机器复现实验，需要另外准备这些样本文件。

当前实验环境为 conda 的 `lyric`（Python 3.11），依赖清单见 [environment.yml](/Users/lamptales/remote/cloudmusic2ktv/lyric_align/environment.yml)。已经安装 `pykakasi`、SudachiPy/UniDic 和 `pyopenjtalk`；基线会优先使用 pykakasi，后续实验可直接比较其他后端。

模型权重和声学中间文件统一放在当前目录的 `models/` 或 `artifacts/`，这两个目录已加入 `.gitignore`，不会进入 Git 提交。

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
