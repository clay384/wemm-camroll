# CamRoll · Qwen3.7-Flash + WeMM

此分支基于上游 `thaoshibe/camroll` 的 `1967f977`，改为云端 Qwen3.7-Flash 生成与现有 WeMM 单 worker 检索。上游说明归档在 [UPSTREAM_README.md](docs/UPSTREAM_README.md)，其中旧模型安装命令不适用于本部署。

本轮已部署并验证，详见 [2026-09-15 验收与检索结果](docs/VALIDATION_20260915.md)。保持 M1 为默认：800题 Hit@1 为80.25%，M3双路为77.50%；四种问答模式已接通，最终答案质量未批量评分。

## 实际链路

照片清单 → 内容哈希与预处理 → Qwen 中文描述/事件 + WeMM 图片编码 → 中文 FTS5 与 WeMM 文本/图片索引 → 五工具循环 → Qwen 回答。

| 配置 | 召回 | 最终回答 |
|---|---|---|
| M1（默认） | WeMM 图片 | Agent 按需调用 view_image |
| M2 | WeMM 图片 | 追加最多6张证据图复核 |
| M3 | WeMM 图片+caption，RRF60 | Agent 按需调用 view_image |
| M4 | WeMM 图片+caption，RRF60 | 追加最多6张证据图复核 |

`search`、`grep`、`list_by_date`、`get`、`view_image` 保留；文件路径由程序维护，模型只操作稳定ID。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-wemm.txt
.venv/bin/pip install --no-deps -e camroll-agent
```

无需在 CamRoll 环境安装 torch、Transformers、FAISS 或下载模型权重。WeMM 由独立服务提供。

配置 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL`、`CAMROLL_BUDGET_DB`、`WEMM_BASE_URL`、`WEMM_KEY_PATH`，详见 [Qwen费用与调用说明](docs/QWEN_RUNTIME.md)。实际服务器配置保存在受限 runtime 文件中，不属于代码交付。

## 使用

清单支持原 CamRoll 的 `turns[].user.image`，或 `images: [{"path":"/absolute/photo.jpg","date":"2026-09-15"}]`。参考照可选，日期缺失时读取EXIF，仍缺失则记为未知。

```bash
camroll-agent inspect album.json
camroll-agent run album.json -o runtime/albums/my-album --resume
camroll-agent enrich album.json -o runtime/albums/my-album --resume
camroll-agent embed-images album.json -o runtime/albums/my-album --resume
camroll-agent ask "照片中有哪些红色汽车？" --memory runtime/albums/my-album --config M1 --json
camroll-agent index runtime/albums/my-album
camroll-agent budget
```

成功描述、编码和事件阶段按版本缓存，重复 run 默认续跑；只有明确需要重做时使用 `--force`。`index` 仅使用已有描述重新建索引，不调用云端生成。每题最多25决策轮、50实际云请求、5次view，所有角色共享本轮300元费用账本。

`enrich` 和 `embed-images` 可分别预建云端描述/事件与图片向量缓存；首次完成后执行 `build --resume` 生成文本向量并发布索引。运行及恢复见 [部署文档](docs/DEPLOYMENT.md)，模块与优化矩阵见 [实现映射](docs/IMPLEMENTATION.md)。

`Agent.ask()` 的 `stopped_reason=ok` 表示正常结束，不等于答案正确率已经评测。

## HTTP

`POST /api/ask` 接收 `album_id`、`question`、`config`（M1–M4），可选 `stream=true`。返回答案、证据ID、工具轨迹、索引版本、停止原因、耗时和usage。`GET /api/health` 返回已发布相册。服务默认只绑定本机 `127.0.0.1:8770`，可通过 SSH 隧道从 Mac 调用。

## 验证

```bash
PYTHONPATH=camroll-agent .venv/bin/python -m pytest -q tests camroll-agent/tests
```

检索评测与云端问答评分分开。本轮仅进行冻结 PhotoBench 图片检索对照和组件/接口验证，不批量评分最终答案。日期、代码与索引指纹随报告保存，历史记录不代表当前服务健康。
