# 日语歌词读音与字级对齐：方案调研与实验路线

本文是 `lyric_align` 实验目录的第一版方案记录。目标不是马上替换生产流水线，而是先找到在网易云句级 LRC、日语歌曲和现有视频渲染器约束下，性价比最高的对齐方法。

## 1. 先把问题拆开

有两个容易混在一起的问题：

1. **文本侧读音**：从歌词原文得到上下文相关的假名，再由假名得到罗马音，并且保留原文字符/词的对应关系。
2. **音频侧时刻**：把读音序列（最好是音素或 mora，而不是 Unicode 字符）和歌曲中的演唱对齐，得到每个可显示单元的 `start_ms/end_ms`。

读音可以在没有音频的情况下生成；时刻对齐则必须面对伴奏、拖长音、连读、重复演唱、空拍和网易云时间戳误差。两者应当分别评估，也应当允许人工修正其中一侧。

## 2. 从现有素材得到的约束

当前示例的 `lyrics_timeline.json` 是句级时间轴，`end_ms` 基本取下一句的开始时间。例如 `椎名林檎/人生は夢だらけ` 有一行跨越约 30 秒的间隔，这不能直接当作该句的演唱时长。时间轴还包含“编曲：…”等非演唱元数据、空行和中英日混排。

已有 `romalrc` 可以作为弱提示，但不能作为真值：有些歌完全没有它；部分汉字读音、数字、英文和专有名词会错误，且其空格只是 mora/音节样式，不含字级时间。

因此第一步应保留原始文本和原始时间戳，另建规范化/对齐产物，不要覆盖网易云文件。

## 3. 文本侧：假名和罗马音

### 推荐的生成链

```text
原文 LRC
  -> Unicode/标点/数字/英文规范化
  -> 日语形态分析（词边界 + 词的读音）
  -> 词内 surface↔reading 对齐
  -> 假名（规范中间表示）
  -> Hepburn 等罗马音
```

候选工具：

- [SudachiPy](https://github.com/WorksApplications/SudachiPy) 或 [fugashi/MeCab](https://github.com/polm/fugashi)：词边界和词典读音；UniDic 对活用词、外来语通常比简单字符替换可靠。
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)：上下文相关 G2P，可作为另一套候选读音和音素序列来源。
- [pykakasi](https://github.com/miurahr/pykakasi)：假名到罗马音；也可在前端用 wanakana 类库，但后端统一生成更容易复现。

假名必须是规范中间层。不要直接从汉字逐字查字典：`今日`、`明日`、`人気`、人名/地名和歌词中的 ateji 都依赖上下文。形态分析器仍会有错误，所以每个词应保存 `reading_source`、候选读音和置信度，并提供歌曲级覆盖表（artist/title/line/token）。

词内对齐不能简单按字符平均切分。应利用 surface 中已有的平假名/片假名锚点，把剩余汉字区间分配给 reading；例如 `読みたい` 至少要得到 `読→よ`、`み→み`、`たい→たい` 的可追溯映射。对于无法唯一分配的汉字，保存一个 token 级 reading，并在 UI 中整体显示假名。

罗马音只作为显示层：选定一种规范（建议 Hepburn，明确长音、撇音、`ん` 的规则），不要把网易云的空格形式当作对齐单位。真正的对齐单位建议是 **mora/音素**；一个汉字可能对应多个 mora，一个小 `っ` 或长音符号可能没有独立可听起点。

特殊处理应显式建模：助词的读法（`は→わ`、`へ→え`）、`っ`、`ん`、长元音、片假名外来语、英文/数字、括号内注音，以及歌手省略或改变的发音。文本显示可以保留原文，音频对齐序列则允许一个 `spoken_form` 覆盖标准读音。

## 4. 音频侧候选路线

### A. 立即可做的基线：锚点内 mora 插值

把每句原文转为 mora，使用网易云句首/下一句句首作为大致窗口，再按 mora 权重分配时长。权重可先全部相等，随后用音符/F0 起点、能量峰和静音检测修正。它不是“唱到哪”的最终方案，但实现简单，能作为所有模型的回归基线，也能在模型失败时降级。

### B. CTC/强制对齐：最值得先做的模型路线

1. 用 [Demucs](https://github.com/facebookresearch/demucs)（或同类 UVR/Spleeter）缓存人声 stem；全混音会让声学模型把伴奏当成噪声。
2. 用日语 wav2vec2/HuBERT 声学模型产生帧级 CTC 后验，把假名转换成音素序列。
3. 用 Viterbi 或 [ctc-segmentation](https://github.com/lumaku/ctc-segmentation) 在每句锚点窗口内求最可能路径。
4. 将音素时间合并回 mora、词和显示字符。

这是可解释、可约束、容易输出置信度的路线。普通语音模型对歌声的音高、延音和混响有明显域差，因此应先在 5--10 首歌上测量错误；必要时再用少量人工标注做 singing-domain 微调。静音/伴奏段、和声和多人声需要单独的 vocal-activity mask。

### C. Whisper 系列：适合粗定位，不宜直接当字级真值

[Whisper](https://github.com/openai/whisper) / [WhisperX](https://github.com/m-bain/whisperX) / [stable-ts](https://github.com/jianfch/stable-ts) 能快速给出句或词级时间，适合校正网易云句首、发现漏词和生成候选转写。它们的 token 不是日语字符或 mora，歌声识别也容易漏字、重复和幻觉；WhisperX 的 wav2vec2 二次对齐同样主要针对语音。可把结果作为 CTC 的先验/异常检测，不建议直接驱动最终扫色。

### D. 专门的歌词/人工标注工具：用作校验闭环

- [Sonic Visualiser](https://www.sonicvisualiser.org/) + [Tony](https://code.soundsoftware.ac.uk/projects/tony)：适合查看波形、频谱、F0 并交互修正歌词边界，可作为早期人工标注工具。
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)：若当前模型仓库有可用日语 acoustic model/dictionary，可作为 GMM/HMM 基线；需用 `mfa model list` 核对版本，不应假定所有发行版都自带日语模型。它原生面向语音，歌声结果仍需验证。
- [DALI](https://github.com/gabolsgabs/DALI) 等公开歌词时间数据可用于验证算法接口；其语言/标注粒度与日语歌曲不完全匹配，不能替代自己的测试集。

## 5. 关键算法设计

### 5.1 分层对齐，而不是一次求字符时间

```text
句（网易云锚点）
  -> 词/短语（形态分析）
  -> mora
  -> 音素/CTC 帧
  -> 显示字符的聚合时间
```

先在较稳定的句窗口内对齐音素，再聚合到 mora/词，最后按照 surface↔reading 映射把时间投影到字符。对一个汉字对应多个 mora 的情况，UI 可以按 mora 扫色，也可以让同一汉字在其全部 mora 完成后变色；这应是渲染选项，而非算法硬编码。

### 5.2 使用句级时间戳作为软约束

将网易云 `start_ms` 作为窗口中心而非硬边界；允许前后扩展（如 ±1--3 秒），用声学后验和 vocal activity 重新估计实际首尾。相邻句、重复句和合唱句需要允许重叠或一对多匹配。元数据行、纯标点、空行先标记为 `non_sung`，不进入音素对齐。

### 5.3 置信度和失败降级

每个层级都保存 `method`、`confidence`、`warnings`。低置信度时依次降级为：音素 CTC → mora 插值 → 句级匀速。这样不会因为一处生僻人名或英文 rap 使整首歌无法生成。

## 6. 建议的实验顺序

### Phase 0：定义数据和评测（半天到 1 天）

选 8--12 首代表性歌曲：清晰独唱、快歌、长音/气声、汉字生僻、片假名/英文、多人声各若干。手工标注句内 mora 或字符边界作为小型金标准。指标使用边界 MAE/中位数、±50/100 ms 命中率、读音准确率，以及实际视频中“提前/滞后”的主观等级。

### Phase 1：文本读音（1--2 天）

实现规范化、Sudachi/MeCab + pyopenjtalk 双候选、词内 surface↔reading 映射、Hepburn 转换和人工覆盖 JSON。先不碰音频，确保 `lyrics_reading.json` 对现有所有 outputs 可复现，并把错误案例收集成回归测试。

### Phase 2：插值基线与可视化（1 天）

实现 mora 拆分、句锚点修正和静音感知插值；输出带中间标记的 ASS/JSON，在播放器或静态图上检查扫色速度。此阶段能迅速判断“句级时间本身是否足够好”。

### Phase 3：人声 + CTC 原型（2--5 天）

加入 Demucs 缓存、日语声学模型和 ctc-segmentation；先只做 1--2 首歌，保存 posterior、对齐路径和失败原因。与 Phase 2 同时输出，比较指标和主观观感。

### Phase 4：人工校正闭环（随后）

做一个最小校正器：播放人声/原曲、显示假名和波形，拖动句/词/mora 边界，导出覆盖文件。人工结果既是产品能力，也是未来 singing-domain 微调数据。

### Phase 5：域适配（有数据后）

当积累约 30--100 首、每首若干分钟的高质量边界后，再考虑微调 CTC 或训练专门的日语歌声对齐模型。没有这批标注前，直接训练大模型的投入/收益通常不如强约束的 CTC + 人工校正。

## 7. 建议的产物格式

不要改写 `lyrics_timeline.json` 的兼容字段；新增例如 `lyrics_alignment.json`：

```json
{
  "version": 1,
  "lines": [{
    "start_ms": 12260,
    "end_ms": 16476,
    "text": "あとすこしそばにいて",
    "status": "aligned",
    "method": "ctc",
    "confidence": 0.82,
    "tokens": [{
      "surface": "あと",
      "reading": "あと",
      "romaji": "ato",
      "start_ms": 12260,
      "end_ms": 12840,
      "mora": [{"text": "あ", "start_ms": 12260, "end_ms": 12520}]
    }]
  }]
}
```

实际实现可把 `chars`、`mora` 和 `phonemes` 分开保存，避免渲染器被迫了解声学模型细节。时间统一使用整数毫秒；所有中间文件（分离后人声、后验、TextGrid/路径）按歌曲目录缓存，便于复盘。

## 8. 初步结论

最现实的路线是“上下文 G2P + 假名为中间层 + 人声分离 + CTC 强制对齐 + 句级锚点软约束 + 人工校正/降级”。WhisperX、MFA 和 Tony/Sonic Visualiser 都值得作为基线或标注工具，但不应把任一普通语音模型的词时间直接当成日语歌声的字时间。先做 Phase 0--2 可以在很低成本下验证数据问题；只有基线明确不够时，才投入 Phase 3 的声学模型和后续域适配。
