# 歌词读音与字级对齐实验流水线

本文定义实验目录的输入/输出约束和内部处理边界。它描述的是当前实验实现以及拟接入原项目的兼容接口；实验目录不会直接修改原仓库的 `outputs/`。

## 1. 目标和非目标

目标是把网易云的句级歌词时间轴转换为可用于 KTV 渲染的层级化时间轴：

```text
原文句 → 词 → mora/音素 → 时间区间
```

并提供日语假名/罗马音。运行时不依赖人工修正；模型低置信度时自动回退到更粗粒度的时间轴。人工页面只用于离线诊断和构建未来评测集。

当前不保证：歌手特殊读法一定被自动识别、和声一定被排除、每个字符都达到人工标注精度。CTC 模型是普通日语语音模型，不是专门的歌声模型。

## 2. 目录约定

```text
lyric_align/
  samples/       # 本机回归样本，完整目录被 .gitignore 忽略
  models/        # 模型和词典缓存，被 .gitignore 忽略
  artifacts/     # 人声 stem、声学中间文件，被 .gitignore 忽略
  results/       # 可再生的 JSON/HTML 报告，被 .gitignore 忽略
  align_baseline.py
  prepare_reading.py
  activity_baseline.py
  ctc_align.py
  ctc_mora.py
```

`samples/<song_dir>/` 至少需要：

- `metadata.json`：用于标题/艺人展示；
- `lyrics_timeline.json`：JSON 数组，每行含 `start_ms`、`end_ms`、`text`；
- 一个 `audio.*` 文件（mp3/wav/m4a 等 FFmpeg 可读格式）。

允许存在 `lyrics.lrc`、封面和原项目其他文件，但实验脚本不依赖它们。歌曲目录名必须在本地文件系统中保持不变，因为报告通过目录名关联音频、歌词和结果。

## 3. 时间和文本不变量

- 所有时间均为整数毫秒，区间采用半开区间 `[start_ms, end_ms)`；
- 必须满足 `0 <= start_ms <= end_ms`；
- 句级 `end_ms` 通常来自下一句的 `start_ms`，只是锚点，不等于真实演唱结束；
- mora/字符区间应当落在其所属句的搜索窗口内，允许为模型 margin 留出窗口外帧，但最终投影必须裁剪回歌曲时间轴；
- 空行、纯标点和元数据行不参与声学对齐；原文仍可在诊断报告中保留；
- 原文显示文本永远不被 reading 覆盖。reading 是附加层，音频对齐使用 reading/音素序列。

## 4. 文本侧处理

### 4.1 后端模式

`prepare_reading.py --backend` 支持：

- `pykakasi`：轻量、适合快速假名和罗马音；
- `sudachi`：提供词边界和 UniDic 读音；
- `openjtalk`：上下文 G2P，并可给出音素；
- `consensus`：离线比较三者，冲突时默认 Sudachi 并标低置信度。

生产式任务只需一个后端。例如：

```bash
conda activate lyric
python prepare_reading.py --backend openjtalk --out results/reading_openjtalk
```

### 4.2 reading 规则

假名是规范中间表示，罗马音只由假名生成。小假名、促音、鼻音和长音不可简单按 Unicode 字符平均处理；内部声学单位优先使用 mora 或 OpenJTalk 音素。

`surface → reading` 不保证逐字符唯一映射。词典可能将 `君` 读成 `きみ` 或其他候选，歌手还可能使用特殊读法。因此每行都要保留 `method`、`confidence` 和 `warnings`。

## 5. 音频侧处理

### 5.1 活动窗口基线

`activity_baseline.py` 使用 FFmpeg 解码为 16 kHz 单声道，再计算约 30 ms 短时 RMS。在句级窗口内寻找活动边界，仅用于缩小后续搜索范围。它不能区分人声和伴奏，结果必须带 warning。

### 5.2 人声分离

Demucs 的模型缓存通过：

```bash
export HF_HOME="$PWD/models/huggingface"
```

输出放在 `artifacts/demucs_<name>/htdemucs/<track>/vocals.wav`。分离结果是中间文件，不覆盖原音频；脚本必须允许缓存复用。

### 5.3 CTC 强制对齐

CTC 模型输入人声波形，输出每个声学帧对词表标签的概率。歌词 reading 已知时，`ctc_align.py` 在句级窗口（当前为前后各 500 ms）内用 Viterbi 搜索一条必须经过完整 reading 的路径。它不是重新识别歌词，而是在已知歌词下估计每个字符的时间。

当前模型：`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`。模型目录必须在 `models/huggingface/`，推理使用 `local_files_only=True`，避免运行时偷偷联网。

## 6. 统一产物格式

拟作为渲染器输入的统一文件名为 `alignment.json`。当前实验脚本分别输出 reading、activity 和 CTC 文件；迁移到生产时应合并成以下结构：

```json
{
  "schema_version": 1,
  "song": {"id": 1867888452, "title": "なつのせいです", "artist": "羊文学"},
  "source": {"timeline": "lyrics_timeline.json", "audio": "audio.mp3"},
  "reading_backend": "openjtalk",
  "lines": [{
    "line_index": 0,
    "text": "それは夏のせいです",
    "reading": "それはなつのせいです",
    "romaji": "sorewa natsu no sei desu",
    "start_ms": 1290,
    "end_ms": 6610,
    "alignment_status": "ctc",
    "confidence": 0.78,
    "method": "demucs+ctc",
    "warnings": [],
    "mora": [{
      "text": "そ",
      "start_ms": 1340,
      "end_ms": 1510,
      "confidence": 0.62,
      "chars": [{"text": "そ", "start_ms": 1340, "end_ms": 1510}]
    }]
  }]
}
```

字段约束：

- `alignment_status`：`ctc`、`activity`、`interpolation`、`unresolved`；
- `method`：可审计的流水线名称；
- `confidence`：`0.0–1.0`，不是跨模型严格可比的概率；
- `warnings`：面向诊断，不得导致手机端任务失败；
- `mora` 按时间递增；相邻区间允许有空隙，不允许反向；
- `chars` 可为空或多个字符，支持一个汉字对应多个 mora；
- 渲染器只依赖 `text`、`start_ms`、`end_ms` 和 `mora`，不需要理解模型 posterior。

## 7. 自动质量门控和回退

运行时按以下顺序选择：

```text
CTC 路径完整、覆盖率足够、分数过阈值
  → 使用 CTC mora 时间

否则存在可靠人声活动窗口
  → 使用活动窗口内插值

否则
  → 使用原句级窗口内插值
```

质量门控必须是确定性的，并把阈值和版本记录在产物中。当前 CTC 探针默认分数阈值为 `-1.5`，这是实验阈值，不应直接视为最终产品阈值。

## 8. 渲染器接入约束

现有 CloudMusic2KTV 仍读取 `lyrics_timeline.json` 和单个 `progress`。未来接入时：

1. `VideoProject.load()` 优先读取同目录 `alignment.json`，不存在时保持旧行为；
2. `_draw_lyric_block()` 保留原文布局和翻译/罗马音布局；
3. `_draw_wipe_text()` 增加按 mora/字符的 mask，未提供细粒度结果时使用句级进度；
4. 对已唱字符采用累积高亮，对当前字符使用更亮颜色；
5. 视频生成不应依赖浏览器或在线模型服务。

## 9. 可重复运行

推荐命令顺序：

```bash
conda activate lyric
python prepare_reading.py --backend openjtalk --out results/reading_openjtalk
python activity_baseline.py
export HF_HOME="$PWD/models/huggingface"
python ctc_align.py --song <song_dir> --vocals <vocals.wav> \
  --reading <reading.json> --out results/ctc_<name>.json
python ctc_mora.py --input results/ctc_<name>.json \
  --out results/ctc_<name>_mora.json
```

所有结果目录都可删除后重建；样本、模型和中间音频由 `.gitignore` 排除，不应写入生产 Git 提交。
