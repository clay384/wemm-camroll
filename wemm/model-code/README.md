# WeMM 模型内部架构源码（不含权重）

本目录补齐 `backend/encoder.py` 在外部模型目录中通过 `trust_remote_code=True` 加载的公开模型实现，供代码阅读和架构审计。它是已锁定模型版本的只读源码副本，**不会改变现有 encoder 的加载路径，也不能独立执行推理**。

来源为 `tencent/WeMM-Embedding-9B`，固定 revision `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`。2026-10-07 从现有部署只读取得的 14 个小型文件，逐项与 [`../SOURCE_LOCK.json`](../SOURCE_LOCK.json) 的大小和 SHA256 一致；没有读取或复制权重、词表或私人数据。完整相对路径、来源 URL 和校验值见 [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json)。

- [`modeling_wemm_embedding.py`](modeling_wemm_embedding.py)：继承 Transformers 的 `Qwen3_5ForConditionalGeneration`；每次 embedding 清理 `rope_deltas`，取最后一个有效位置的隐藏状态并归一化。
- [`modeling_st_wemm.py`](modeling_st_wemm.py)：Sentence Transformers 的多模态输入适配。
- [`config.json`](config.json) 与 [`processor_config.json`](processor_config.json)：模型结构、自动加载映射及图像/视频处理配置。
- `*.jinja` 与 `additional_chat_templates/`：原版对话及 embedding 输入模板。
- 其余小型 JSON 和 `patch_sglang_video.py`：上游 Sentence Transformers / SGLang 入口材料，未接管当前 CamRoll 推理链路。
- [`UPSTREAM_MODEL_CARD.md`](UPSTREAM_MODEL_CARD.md)：原 `README.md` 字节原样保留，仅本仓库文件名改变。
- [`LICENSE`](LICENSE)：模型仓库原许可证，包含第三方组件条款；不要用应用层许可替代它。

实际部署仍需另外取得锁定权重、`tokenizer.json`、匹配的依赖环境，以及本地配置。见 [`../../docs/MODEL_DEPENDENCIES.md`](../../docs/MODEL_DEPENDENCIES.md)。`SOURCE_MANIFEST.json` 仅覆盖取得的上游文件，不将本说明或自身纳入循环哈希。
