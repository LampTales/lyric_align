# 歌词读音与字级对齐实验流水线

本文定义当前库的输入/输出约束和内部处理边界；历史方案调研见 [docs/DESIGN_RESEARCH.md](docs/DESIGN_RESEARCH.md)，近期 offset 现象见 [docs/OFFSET_RESEARCH_NOTES.md](docs/OFFSET_RESEARCH_NOTES.md)。它描述的是当前实验实现以及拟接入原项目的兼容接口；实验目录不会直接修改原仓库的 `outputs/`。

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
  src/lyric_align/    # installable library
  tests/              # API and schema regression tests
  results/            # optional local reports (ignored)
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
- 网易云句首可能存在整首歌曲级偏移。偏移修正应作为独立预处理步骤，输出 `global_offset_ms` 并生成修正后的句级锚点；不能把它混入单句 CTC 的局部 margin。

## 4. 文本侧处理

### 4.1 后端模式

库的 `AlignmentConfig.g2p_backend`（以及 CLI 的 `--g2p-backend`）支持，默认使用
`sudachi`：

- `pykakasi`：轻量、适合快速假名和罗马音；
- `sudachi`：提供词边界和 UniDic 读音；
- `openjtalk`：上下文 G2P，并可给出音素；

根目录的旧实验脚本 `prepare_reading.py` 另支持 `consensus`，用于离线比较三种后端；
它不是安装库的 `AlignmentConfig` 选项，也不属于生产 API。

生产式任务只需一个后端。例如：

```bash
conda activate lyric
python prepare_reading.py --backend sudachi --out results/reading_sudachi
```

### 4.2 reading 规则

假名是规范中间表示，罗马音只由假名生成。小假名、促音、鼻音和长音不可简单按 Unicode 字符平均处理；内部声学单位优先使用 mora 或 OpenJTalk 音素。

`surface → reading` 不保证逐字符唯一映射。词典可能将 `君` 读成 `きみ` 或其他候选，歌手还可能使用特殊读法。因此每行都要保留 `method`、`confidence` 和 `warnings`。

声学 `tokens` 是 CTC reading 字符，`mora[].chars` 是 reading 字符，并**不是**原文字符到假名的映射。原文注音使用独立的 `surface_spans`（由库自动生成）：

```json
"surface_spans": [
  {"surface": "夏", "surface_start": 3, "surface_end": 4,
   "reading": "なつ", "reading_start": 3, "reading_end": 5,
   "romaji": "natsu", "mora_indices": [3, 4]}
]
```

`surface_start/end` 和 `reading_start/end` 使用 Python 字符串索引（不是 UTF-8 字节偏移）；标点和空白可有 span 但不参与音频对齐。一个 surface 字符可以覆盖多个 reading/mora，一个 reading mora 也可以对应多个 surface 字符；无法唯一分配时应使用词级 span 并标记 `mapping_confidence=low`。渲染器可以据此在原文字符上方绘制假名，或按同一 span 聚合罗马音。

## 5. 音频侧处理

### 5.0 全局偏移估计

`offset_search.py` 在默认 ±2000 ms 范围内扫描候选偏移，比较歌词句首与音频局部能量起始的统计一致性。它只提供候选偏移，不能证明句首一定是人声；应优先使用人声 stem，或在 CTC 句级分数上做二次验证。最终产物保存原始和修正后的时间：

```json
"timing": {"original_start_ms": 33330, "global_offset_ms": -420, "start_ms": 32910}
```

当前库默认增加人声边界约束：长静音至少 2000 ms，
持续活动至少 200 ms，边界容差 ±800 ms。开头只绑定第一句；间奏还要求原始
歌词区间之间有明确长空隙且附近候选句唯一。一句内部的长停顿或关联不明确的
边界跳过，约束冲突则保留原时间轴，不强行选择新偏移。没有可靠边界时沿用
能量评分；原混音不施加这项人声约束。该步骤不证明声音对应哪个歌词。

`--offset-acoustic-verify` 可显式开启实验性 CTC 验证，默认关闭。最多比较
3 个偏移 × 3 句固定歌词，只决定接受或否决能量候选，不修改字级对齐。
证据不足回到 0；缺模型或人声时报错。实测它会误拒田中愛愛愛子正确的
+880 ms，因此目前不建议默认启用。参数、诊断字段和缓存语义详见 `LIBRARY.md`。

只有当最佳候选相对 0 ms 的增益至少为 `0.08`、相对相邻搜索点的局部峰值突出度至少为
`0.01`，且没有落在搜索边界时才自动应用；否则保留
`global_offset_ms=0` 并标记 `offset_status="uncertain"`。这是面向无人值守 KTV
队列的保守启发式门控，不是统计学显著性检验。

`apply_offset.py` 可把候选偏移写入独立的修正时间轴，永远不覆盖 `lyrics_timeline.json`：

```bash
python apply_offset.py \
  --timeline samples/1372726250_サカナクション_ユリイカ/lyrics_timeline.json \
  --offset-ms -440 \
  --out results/offset_yuriyika_timeline.json
```

正值表示歌词整体向后，负值表示整体向前。修正文件保留 `original_start_ms/end_ms`，渲染器只读取修正后的 `start_ms/end_ms`。

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

库直接生成作为渲染器输入的统一文件 `alignment.json`：

```json
{
  "schema_version": 1,
  "song": {"id": 1867888452, "title": "なつのせいです", "artist": "羊文学"},
  "source": {"timeline": "lyrics_timeline.json", "audio": "audio.mp3"},
  "reading_backend": "sudachi",
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
- `surface_spans`（注音渲染必需）由 `build_reading_lines()` 生成，应覆盖原文中需要注音的字符；它与声学 `tokens` 是不同层级，不能互相替代；
- 渲染器只依赖 `text`、`start_ms`、`end_ms` 和最终 `display_units`，不需要
  理解模型 posterior，也不应再次按 `singing_end_ms` 缩放 token/mora。
- `display_units` 中没有 CTC/mora 锚点的可见字符（如标点、模型词表中
  被省略的拉丁片段）会被放入最近两个已对齐字符之间的局部空白，而不是
  按整句比例插值，避免侵入相邻 mora。若多个表面字符继承同一 mora 的
  时间，或仍存在异常重叠，写入产物时再按原有起点（相同起点时按时长
  权重）拆成顺序区间。
  空白字符仍作为显式的时间边界，不参与该拆分。

## 7. 自动质量门控和回退

运行时按以下顺序选择：

```text
CTC 路径完整、覆盖率足够、分数过阈值（零时长边界先做可审计的显示修复）
  → 使用 CTC mora 时间

否则存在可靠人声活动窗口
  → 使用活动窗口内插值

否则
  → 使用原句级窗口内插值
```

质量门控必须是确定性的，并把阈值和处理策略记录在产物中。当前 CTC 探针默认分数阈值为 `-1.5`，这是实验阈值，不应直接视为最终产品阈值。

## 8. 渲染器接入约束

现有 CloudMusic2KTV 仍读取 `lyrics_timeline.json` 和单个 `progress`。未来接入时：

1. `VideoProject.load()` 优先读取同目录 `alignment.json`，不存在时保持旧行为；
2. `_draw_lyric_block()` 保留原文布局和翻译/罗马音布局；
3. `_draw_wipe_text()` 增加按 mora/字符的 mask，未提供细粒度结果时使用句级进度；
4. 对已唱字符采用累积高亮，对当前字符使用更亮颜色；
5. 视频生成不应依赖浏览器或在线模型服务。

## 9. 服务形态和 Docker

推荐把本项目实现成**可嵌入的 Python 库 + 可选 CLI**，而不是每次生成视频都通过 HTTP 调用：

- 库 API 接受本地音频路径、句级 timeline 和配置，返回内存中的 `AlignmentResult`，并可写出 `alignment.json`；
- CLI 负责批处理、缓存和日志；
- 现有 CloudMusic2KTV 后端在同一 Python 运行时内直接调用库，避免上传/下载音频和 JSON 的额外延迟；
- 如果未来需要独立 GPU 节点，再在库外增加异步 HTTP worker，API 仍应传递版本化 `alignment.json`，而不是让视频渲染器依赖模型服务在线可用。

正式 Docker 部署建议：

```text
backend image
  ├─ Python 代码 + FFmpeg + 字体
  ├─ 不包含模型权重
  └─ 挂载 /var/lib/lyric-models → models/
```

通过环境变量配置：

- `LYRIC_MODEL_DIR=/var/lib/lyric-models`；
- `HF_HOME=/var/lib/lyric-models/huggingface`；
- `LYRIC_G2P_BACKEND=sudachi`；
- `LYRIC_DEVICE=cpu` 或 `cuda`（若节点有 GPU）。

容器启动时使用 `local_files_only=true`；模型缺失应在健康检查/任务状态中报告明确错误，不应在视频请求期间偷偷联网下载。模型目录作为独立 Docker volume 或宿主机只读 bind mount 配置，升级镜像不改模型；模型版本写入 `alignment.json`。Demucs 和 wav2vec2 权重均不进入镜像层。Sudachi 及其系统词典属于 Python 包依赖；实验性的 pyopenjtalk 后端需要单独安装。

## 10. 耗时基线（Apple Silicon MacBook，CPU）

以下是当前环境的实测量级，歌曲分别为《なつのせいです》（约 234 秒）和 19 首样本批处理；实际时间会随 CPU、线程和缓存变化：

| 步骤 | 实测量级 | 是否可缓存 |
|---|---:|---|
| 单后端 G2P（19 首、653 行） | 约 1.2 s | 是，按歌词 hash |
| 全样本短时能量（19 首） | 约 4.4 s | 是，按音频 hash |
| Demucs 人声分离（234 s 歌曲） | 约 52 s CPU | 是，按模型+音频 hash |
| wav2vec2 CTC（8 句） | 约 4.8 s（含模型加载） | 模型常驻后更快 |
| wav2vec2 CTC（整首 36 句） | 约 10–30 s 量级 | 是，可按句窗口缓存 |

耗时大头是 Demucs，其次是 CTC 推理；G2P 和能量分析可以忽略不计。当前库 API
会在进程内缓存 Demucs 和 CTC 模型；修改模型文件或 device 后调用
`lyric_align.clear_model_cache()`。服务化时仍建议让模型常驻进程并批量处理句窗口。

## 11. 混合语言和非演唱行

- 英文/数字：当前 reading 保留粗粒度 ASCII 单元；日语 CTC 词表对英文、URL、缩写和数字不一定有标签，需将这些 span 标记为 `unresolved` 或交给英文声学模型，不能强行映射成日语读音；
- 片假名外来语：通常可由 OpenJTalk/Sudachi 处理，但歌手按英文发音演唱时可能偏离；
- 括号注音（如 `海月（くらげ）`）：应优先使用显式 ruby，并从声学目标中去除括号内容重复；
- 元数据、标题、作词/编曲：文本检测只能作为初筛。若句级窗口中人声活动比例接近 0，可标记 `non_sung`；但有伴奏、和声或呼吸时不能仅凭能量证明“有人声”。最终应综合 vocal-stem 活动、CTC 路径可行性和句间上下文；
- 纯 instrumental/间奏：保留原行供显示审计，但不生成 mora，不参与 CTC。

## 12. 可重复运行

推荐使用库或 CLI 一次完成所需阶段：

```bash
PYTHONPATH=src python -m lyric_align.cli prepare \
  --song-dir samples/<song> --stages reading demucs ctc \
  --demucs-model-path /models/demucs/<snapshot> \
  --ctc-model-path /models/wav2vec2-japanese
```

阶段结果会写入 `alignment.json` 和 `preprocessing.json`。Demucs 的压缩 stem 可独立复用；CTC 失败时下次从句级窗口重新计算，不保存帧级 checkpoint。

旧版实验脚本仍可用于复现实验报告，但不属于公开库 API：

```bash
conda activate lyric
python prepare_reading.py --backend sudachi --out results/reading_sudachi
python activity_baseline.py
export HF_HOME="$PWD/models/huggingface"
python ctc_align.py --song <song_dir> --vocals <vocals.wav> \
  --reading <reading.json> --out results/ctc_<name>.json
python ctc_mora.py --input results/ctc_<name>.json \
  --out results/ctc_<name>_mora.json
```

所有结果目录都可删除后重建；样本、模型和中间音频由 `.gitignore` 排除，不应写入生产 Git 提交。
