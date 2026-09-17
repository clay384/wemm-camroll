# WeMM–CamRoll 相册问答

这是固定 PhotoBench album3 完整版（355 图）的独立局域网页面。它固定使用相册 `eval-20260916-141627-camroll-e5a5fe2f`、索引 `bd64fa387f0c5d507ef845fb2493eed6`、M4 双路检索和 `qwen3.7-flash`，不提供切换或重建入口。

数据流是 `浏览器 → 8773 → 127.0.0.1:8772/camroll/api/ask → CamRoll Agent / Qwen / WeMM → 冻结索引`。8773 对相册与索引文件只读，不持有云密钥、不建第二套费用账本、不调用 Judge；问答仍会由 8772 发起云调用并写入已有累计预算账本。页面没有登录，同源与 Host 白名单不是身份认证。

## 接口

- `GET /api/health`：8773、上游、固定策略和冻结快照状态。
- `GET /api/albums`：唯一相册的公开元数据，不返回服务器路径。
- `POST /api/chat/stream`：只接受 `album_id` 与 `question`，服务端强制 M4、流式和结构化输出。SSE 事件为 `request_started`、`model_status`、`tool_call`、`tool_result`、`answer`、`error`。
- `GET /api/albums/{album_id}/media/{image_id}?variant=thumb|full`：按稳定图片 ID 读取相册内资产；缩略图只在内存生成。

`selected_image_ids` 是最终回答明确采用的图片，`retrieved_image_ids` 是工具候选，两组在页面去重后分别标注。事件级日期、地点、人物和描述会传播到对应图片的预览元数据。浏览器断线不会取消已转交的上游调用，页面不自动重试；全局已有问题执行时，新提交返回 429。

## 安全和完整性

服务验证固定 Host、同源写请求、严格 CSP，并删除系统提示词、模型原始输入输出、密钥和服务器路径。快照的 `records.json`、`search.sqlite`、`vectors.npz` 使用固定 SHA256；每张被读取的原图也必须与其内容寻址 `asset_id` 相符。未知 ID、越界路径、跨相册读取或完整性变化统一阻断。常见状态码包括 404（未知媒体）、409（索引或快照变化）、422（请求无效）、429（已有在途问题）和 503（冻结相册不可用）。

## 本地验证

```bash
PYTHONPATH=camroll/camroll-chat python -m pytest -q \
  camroll/camroll-chat/tests/test_backend.py \
  deployment/test_chat_deployment.py
node --check camroll/camroll-chat/static/app.js
```

生产部署、进程身份校验及回滚见 [`deployment/CHAT.md`](../../deployment/CHAT.md)。
