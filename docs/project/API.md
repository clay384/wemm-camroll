# HTTP 接口参考

依据本地 `backend/app.py`、`retrieval.py`，整理日期 2026-09-15。地址为历史部署地址 `http://192.168.0.100:8765`。FastAPI 禁用了自动 OpenAPI/docs/redoc 页面。

**当前禁止对 `.100` 写入，本页仅供交接，不代表本轮已调用接口。** 搜索也会写事件，验收/评测接口会写报告。

## 认证与通用约定

`POST /api/login` 接收 `{"password":"<从受限文件读取>"}`，成功返回 `{"ok":true}` 并设置 `wemm_session` cookie，24 小时有效，HttpOnly、SameSite=strict。其余 `/api/*` 需要 cookie。POST 若携带 Origin，必须与服务源一致，否则 403。错误密码返回 401。

不要在源码中硬编码口令。访问缩略图、原图同样需要已登录 Session。正常 JSON 错误通常使用 detail；客户端应检查 HTTP 状态及响应内的逐文件错误，不能只看请求是否完成。

## 路由总表

| 方法与路径 | 参数 / 返回 | 行为与副作用 |
|---|---|---|
| POST /api/login | password → ok、cookie | 登录 |
| GET /api/status | 无 → 模型状态、GPU、counts、units、queue、fingerprint 等 | counts/units 是全局统计，不是当前数据集 |
| GET /api/datasets | 无 → 数据集数组 id/name/paused | 列表 |
| POST /api/datasets | name → id/name | 创建，名称去空白、截取 100 字符，空名称 400 |
| POST /api/upload | query: dataset、name；body: 原始文件字节 | 保存文件并排队；**不是 multipart** |
| POST /api/import-directory | JSON dataset、path | 导入 import 根目录内文件，返回 files 数组 |
| POST /api/joint-candidate | dataset、image_asset、text | 创建图片+文本联合候选；本轮 PhotoBench 未使用 |
| GET /api/assets | query dataset，默认 personal | 资产列表，含状态、错误、name、units |
| POST /api/control | dataset、action | pause/resume/retry/cancel，改变队列或状态 |
| GET /api/units/{uid}/{mode} | mode=thumbnail 或 content | 派生内容文件；不存在 404，非法访问 403 |
| GET /api/assets/{asset}/original | SHA256 asset | 原始字节文件 |
| POST /api/search | 见下文 | 查询编码与检索，**写 SQLite 查询事件** |
| POST /api/annotations | query、unit、label | label=relevant/irrelevant/unjudged；保存标注 |
| POST /api/benchmark | 无 | 启动旧 demo 评测并写根 reports；不适用于 PhotoBench |
| GET /api/benchmark | 无 → job、metrics | job 是进程内状态；metrics 可为历史结果 |
| POST /api/validate-alignment | 验证使用固定 smoke 数据 | 写 service_alignment.json / .npz，可能覆盖旧验证 |
| GET /api/report/{name} | 仅允许旧报告名单 | 不能任意读取报告目录 |

前端在 `/` 静态提供。没有通用 `/v1/embeddings`、裸向量导出、答案生成或 router 接口。

## 查询

```json
{
  "dataset": "ad8f153d865a4d0caa5d1704b0874426",
  "text": "在停车场的照片",
  "kind": "image",
  "k": 50,
  "dimension": 4096,
  "aggregate": true
}
```

| 字段 | 默认与说明 |
|---|---|
| dataset | 默认 demo；务必显式指定，以免检索错误候选集 |
| text | 文本问题；与 image_asset 至少提供一种，可同时提供 |
| image_asset | 已有图片资产 ID；必须已产生 image 单元；不是图片路径或上传字节 |
| exclude_self | 使用图片查询时默认 true，排除查询图片自身 |
| kind | 可省略/null：所有类型；image/video/pdf/text 等按字符串筛选，非法字符串可能只返回空结果 |
| k | 默认 10；仅支持 5、10、20、50，不支持 1；Hit@1 从返回结果第一项计算 |
| dimension | 默认 4096；支持 4096、1024、256，截断后重新归一化 |
| aggregate | 默认 true；按源资产最高单元分数聚合 |

响应结构示意（字段值仅为示意）：

```json
{
  "id": "query-id",
  "results": [{
    "id": "unit-id", "asset": "sha256", "kind": "image",
    "name": "photo.jpg", "score": 0.42, "rank": 1,
    "meta": {}, "thumbnail": "/api/units/unit-id/thumbnail",
    "content": "/api/units/unit-id/content", "children": []
  }],
  "timings": {"embedding_s": 0.07, "retrieval_s": 0.01},
  "notice": "相似度是排序分数，不是正确率；Top-K 不保证存在相关结果。"
}
```

timings 还可能含 preprocess_s、queue_s、response_total_s、fingerprint、units_searched、dimension、aggregation 等。分数是余弦排序分数，不能解释为概率或准确率。视频检索结果的 content 指向派生片段；获取源视频可用 original 接口。

## 上传、导入与控制

上传：`POST /api/upload?dataset=<id>&name=<relative-name>`，流式发送文件字节。name 禁止绝对路径、`..`、反斜杠、NUL 等。响应含 asset、duplicate、status、bytes、name。相同内容复用资产，旧失败/取消资产不会因为再次上传就自动重试。

- 空文件或非法名称通常 400；超限 413；队列满 429。
- 目录导入 path 必须在项目 import/ 下；禁止路径逃逸，符号链接记为错误。
- 目录导入最多扫描 500 个文件；HTTP 200 不保证全部成功，要逐项检查 files[].error。
- 不要导入包含 QA/元数据的父目录：导入器会注册文件，格式错误可能在后续处理才失败。
- control.retry 把 failed/cancelled 置为 pending；cancel 作用于未完成资产；pause/resume 控制数据集调度，不杀模型进程。
- joint-candidate 的 text 必须非空且不超过 2000 字符；不是本轮纯媒体索引的一部分。

## 状态与报告

status 主要字段：state、error、model、revision、precision、dimension、offload、profile、limits、native_video、gpu、counts、units、queue、busy_asset、fingerprint、cold_load_s、encoding_calls、benchmark。不能仅凭 state=ready 判断指定数据集全部入库，应再核对其资产状态。

GET /api/report/{name} 只允许：validation_report.html、validation_report.md、metrics.json、results.csv、per_query_results.jsonl、smoke.json。PhotoBench 的 report.html **不在该白名单**，应打开 Mac 本地交付报告。

API benchmark 在任务运行、模型未就绪或 demo 不完整时可返回 409。意外输入或内部异常仍可能产生其他 4xx/5xx，客户端必须保留错误，不应默默跳过。
