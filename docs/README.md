# lyric-align 文档

这里集中维护设计、实验和研究记录；根目录的两个用户文档仍作为公开入口。

## 文档分工

- [README.md](../README.md)：安装、快速使用、Python/CLI 入门和开发命令。
- [LIBRARY.md](../LIBRARY.md)：稳定的库契约，包括输入、输出 schema、配置、阶段缓存和调用方约束。代码行为变化时必须同步更新。
- [PIPELINE.md](../PIPELINE.md)：实现流程说明，包括读音、offset、Demucs、CTC、回退和部署边界。用于理解内部处理顺序，不替代 API 契约。
- [DESIGN_RESEARCH.md](DESIGN_RESEARCH.md)：较早的方案调研、候选技术和实验路线。它记录设计背景，不代表所有建议都已实现。
- [OFFSET_RESEARCH_NOTES.md](OFFSET_RESEARCH_NOTES.md)：2026-09 offset 实验记录和后续研究问题。它记录当前实现的已知现象，不是新的配置规范。
- [../CHANGELOG.md](../CHANGELOG.md)：当前开发线的变更摘要；项目尚未按正式版本发布维护。

## 维护规则

1. 对外可用行为写入 `LIBRARY.md`；实现顺序和内部策略写入 `PIPELINE.md`。
2. 实验观察、样本数据和未决问题写入 `docs/` 下对应研究记录，不把实验结论伪装成稳定保证。
3. 每次修改配置、产物字段、缓存语义或 CLI 参数时，同时更新 README、LIBRARY、PIPELINE 中受影响的入口，并补充测试。
4. `results/`、`samples/`、`models/`、`artifacts/` 和 `temp/` 是本地生成目录，不是发布文档；需要保留的结论必须提炼到本目录。
5. 文档中的时间、阈值和默认值以 `AlignmentConfig` 与 CLI 为准；发现不一致时先修文档或代码，再提交。
