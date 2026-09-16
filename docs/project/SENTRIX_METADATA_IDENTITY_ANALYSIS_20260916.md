# 时间、地点与人物能力：Sentrix 源码核对及 CamRoll 扩展建议

观测日期：2026-09-16。按用户授权只读检查 `.153` 的 Sentrix 源码、PhotoBench 文件与 EXIF，并静态阅读本地 CamRoll。未实例化业务 backend、执行人脸模型、写远端数据库、调用问答、控制进程或启动评测。下文的新工具均为设计建议，未实现。

## 1. 已验证的数据来源

目标目录：`.153:/home/asus/Github/Sentrix-Home-Web/services/photobench/data/album3-max/`。

使用 Pillow 只读遍历 manifest 的全部 355 张照片，结果：

| 项目 | 数量 |
|---|---:|
| 可读取图片 | 355/355 |
| 存在 EXIF | 355/355 |
| 存在拍摄时间标签 | 355/355 |
| 存在 GPS 经纬度标签 | 355/355 |
| 检查到时区偏移标签 | 0/355 |
| 读取异常 | 0 |

这是当前 benchmark 文件中标签存在性的实测，不等于独立验证了每个标签的真实性或其是否由原相机原生写入。时间应保留其原始本地时间语义，缺失时区不能自动视为 UTC，也不能把文件复制/导入时间当拍摄时间。

`album3-max` 当前非媒体文件列表只有 manifest 和 QA；没有该目录自己的 `metadata/image_metadata.jsonl`。已知 JSON 位于混合版：

`album3-max-video10/metadata/image_metadata.jsonl`

它有 334 条记录，包含 capture_datetime、GPS、readable_location、people 及 provenance。与 355 图集交集为 334 图，334 条拍摄日期与图片 EXIF 日期一致。334 条都有地点，159 条有非空 people。该 people 字段的来源说明是 PhotoBench 照片级核心身份标注，不是本次人脸算法的输出。地点来源说明为 OpenStreetMap Nominatim；Sentrix 当前 geocoder 使用的来源不同，不能默认它们精度等价。21 张额外图片不能从这个 334 条 sidecar 自动补齐。

当前 Benchmark `_phase_photo_import` 将原图字节和 scope/batch 信息交给 `/api/import`，没有读取上述 sidecar，也没有传逐图 metadata JSON。Sentrix 的通用 `/api/import` 虽然支持 metadata 参数，当前编排器没有用该能力注入 sidecar。

8 张命名参考照与 6 条家庭关系则由 manifest 显式提供，Benchmark 分别调用 `/api/people/seed-batch` 和 `/api/relationships/batch` 导入。

## 2. Sentrix 的真实处理链

源码根：`.153:/home/asus/Github/Sentrix-Home-Web/`。下列行号为本次观测，不保证后续版本不变。

### 时间与位置

1. `backend/app.py:3535` 的 `/api/import` 接收图片；可选逐文件 metadata 经 `:119` 校验。
2. `backend/pipeline.py:127` 附近的 `prepare_asset` 读取原文件 EXIF，写入 `captured_at`、GPS、设备等字段。`_extract_exif` 在 `:251`。
3. `backend/geocoding.py` 将 GPS 反查地名：中国优先 PyGeoCN/Tianditu，国际回退 reverse_geocoder/GeoNames。来源与精度随结果保存，缺依赖可返回空。
4. `backend/agent_runtime/tools.py:546` 的 `_preview_entry` 直接返回 `captured_at`、`place`、描述及人物投影。`:1837` 的地点展示使用 reverse_geocode 的 city/district 或 label。
5. `backend/structured_memory.py` 使用 SQLite 的拍摄时间、位置、观察记录和人物关联做精确筛选/聚合，而非从向量 Top-K 估计数量。

能力边界：区县/城市反地理编码不能自动证明具体餐馆、景区入口或某个公园；这类 POI 仍需其他可信元数据或视觉/文字证据。代码某些路径存在 file_time 回退，CamRoll 不应将该回退不加标注地当真实拍摄日期。

### 人脸、身份和关系

1. `backend/app.py:2314` 接收命名参考照，调用检测并选择人脸，交给 `backend/db.py:5334` 的 `seed_person_identity`。
2. 预置创建 confirmed 人物实体及人脸簇，保存名字、别名、角色；家庭关系另存。
3. `backend/model_clients.py:1597` 的 `FaceAdapter`，当前 detect 路径调用 RetinaFace 分块检测、buffalo_l/SCRFD 二次校验和人脸 embedding。代码中虽存在 AdaFace 等其他适配文件，不能因此宣称本路径使用 AdaFace。
4. `backend/db.py:4806` 的 `add_face_instance` 按相册隔离，保存 bbox、质量、有效性、embedding 模型版本，再与簇/原型匹配；不合格人脸保留为不具备身份匹配资格的记录。另有 `backend/face_clustering.py` 和重聚类流程。
5. 查询时联结 `face_instances → face_clusters → entities`；家庭关系/归属来自已保存实体关系，不通过外貌猜测。

这里只确认实现结构，没有运行模型或证明这批照片的身份识别率。检测到人脸、与某个簇相似、绑定到命名参考身份、能够看清衣服属性是不同步骤。

## 3. 与本问题相关的 Sentrix 工具

定义在 `backend/agent_runtime/tools.py:2681` 的注册区；通用 ToolSpec 位于 `tool_registry.py`，包含参数、读写类型、scope、超时、成本、readiness、能提供/不能证明的证据类型。

| 工具 | 输入概要 | 真正的数据来源/能力 |
|---|---|---|
| search_memories | query、mode、filters.time/place/person | 检索/硬筛选，预览携带拍摄时间、地点等 |
| query_photo_people | 当前结果中的 asset_handle、result_set_id | 读取单张图片对应的人脸簇与人物实体，返回已绑定姓名、未知同行者、证据资产 |
| inspect_photo | asset_handle、question、target_person | 原图及目标人脸对应的扩展人物裁剪，带逐脸清单看视觉细节 |
| query_memory_facts | operation=count/exists/first/last/group，subject，filters | 对当前相册原始资产、命名人物出现记录等做精确统计；拒绝物体颜色等视觉条件 |
| get_person_profile | person | 已保存人物画像、关系等；不够则返回 limited |

`query_photo_people` 在 `:2548`；`:2270`/`:2313` 读取身份和逐脸清单；`inspect_photo` 在 `:2116`。后者将人物实体对应 bbox 用于局部图像，再由 VLM 回答“这个人穿什么”等问题。这是单纯把 `people=[名字]` 放进 caption 所不能替代的能力。

实现核对发现一处不能盲抄的契约差异：`_confirmed_photo_identities` 返回 `identity_status="bound"`，而 `_preview_entry` 仅接受 `"confirmed"`；前者 SQL 也未显式要求 cluster/entity 状态均 confirmed，另一份 face manifest 则显式检查。它提示身份状态需要统一验证，但本次未执行运行时复现，未作远端修改，也不把函数名当成质量保证。

## 4. CamRoll 当前并非完全没有元数据工具

本地根：`/Users/zhanyiqi/Documents/camwwem/camroll/camroll-agent/camroll_agent/`。

- `tools.py` / `schemas.py` 已有 `list_by_date(date_from,date_to,location,person,...)`。
- `library.py:221` 当前 location/person 是字符串包含匹配，不是规范地点 ID 或命名人物实体查询。
- `search` 与 `grep` 支持日期条件；检索候选在排序前按日期过滤。
- `get` 返回照片/事件记录，可承载来源元数据。
- `pipeline.py:_prepare` 读取 EXIF 日期但仅保存到天，不解析 GPS、时区或精确时间。
- `pipeline.py:_caption` 可接收可信 location/people，但未知时使用模型描述；当前 `people` 混有“人物外观描述”和潜在名字语义。
- `view_image` 接收完整图片和记录，没有逐脸命名定位与目标人物裁剪。

所以主要缺口是**数据提取、来源区分、身份处理和精确查询**，不是缺少工具调用框架。

100 题现有 angle 标签有 place 18、time 15，说明元数据能力直接相关；这不是严格的全部依赖题计数，其他题也可能需要时间/人物。所有题 `required_evidence_sources`/claim support_type 都统一为 event_summary，不能用这个粗标签推断它们不需要元数据或逐脸证据，也不能把人工 event_summary 导入系统。

## 5. 建议：扩展五工具，新增第六个 query_photo_people

### 先建立基础数据

从原图读取并保留原始 EXIF 时间、精度、未知时区、GPS；用明确版本和来源的 geocoder 建立地名与层级，区分 city/district/POI 精度。视觉推测地点和可信 GPS 地点分字段保存。

新增相册级 person、alias、relationship、face_instance、identity_match 数据；逐脸保存图片 ID、bbox、坐标系/EXIF方向、检测与匹配版本、相似度、状态、来源。名字应从命名参考资料或明确人工确认来，不从 caption 或 QA 来。

8 个身份的初版可先做“检测每张图片中的人脸 → 与命名参考原型比对 → 不确定保持 unknown”。不必先实现完整开放集人物聚类与人物画像。若未来需要未知人物跨图归并，再增加匿名聚类，不能把匿名簇直接命名。

### 扩展现有工具

| 工具 | 扩展建议 |
|---|---|
| search | 增加规范 place/person 条件；在向量排序前筛选；结果携带可信时间、地点、身份摘要及来源 |
| list_by_date | 扩展为可靠结构化筛选，支持多个人同时出现、规范地名、分页、总数和覆盖说明；保持旧调用兼容 |
| get | 返回完整时间/GPS/地名来源、身份匹配状态、引用证据 |
| view_image | 增加 target_person_id/face_id，从已验证 bbox 获取人物区域＋原图，回答衣服/动作；定位失败要明确返回 |
| grep | 继续负责文本关键词，不承担身份认证或准确时间推断 |

### 新增 query_photo_people（建议契约，未实现）

输入：`image_ids`（受限数量、来自当前相册的真实 ID），可选 `person_id`。

输出：各图的 `face_id`、`bbox`、`person_id/name`、`identity_status`、`match_score`、`source`、相关可信关系、未知人脸/未处理状态、证据 image_id、处理版本。

scope 和快照由服务注入；工具只读，不接受任意磁盘路径、SQL 或模型自行更改身份绑定。查询返回已计算匹配，而非每次临时把所有图片交给 VLM 猜名字。相似度是算法分数，不冒充准确率或经校准的概率。

不强制新增“读 metadata JSON”工具：记录应先进入经过验证的结构化存储，由 `get` 和筛选工具提供必要字段。直接把整份 JSON 发给模型会混合来源和标注，还会增加上下文。

若后续需要稳定回答“某人第一次出现/共多少张/按月统计”，再加第七工具 `query_memory_facts`，借鉴 Sentrix 的受限精确查询；不能把语义 Top-K 的数量当作全相册计数。

## 6. 典型证据链与边界

- “那次活动是哪天”：语义检索找到活动照片 → get 读取 EXIF 时间 → 用实际精度回答；不依赖模型从画面猜日期。
- “某次活动在哪里”：找到照片 → 读取 GPS 反查地名 → 若问题要求具体 POI，再用视觉文字或合法地点记录验证；城市级信息不足则明确说明。
- “明明穿什么颜色”：找到候选 → query_photo_people 确定这一图中的明明及 bbox → view_image 看其区域与全图 → 引用该照片。仅知道同图包含明明，仍不足以把其中某件衣服归给他。
- “没有查到明明”：必须区分身份没登记、照片未处理、匹配不确定、筛选零结果；不能统一解释为明明不存在或没参加活动。
- “和爸爸一起”：先从明确关系表解析爸爸实体，再筛人物共同出现；亲子/配偶关系不能由长相相似或站位推断。

## 7. 两种实验必须分开

1. **独立端到端系统比较（建议主实验）**：双方接收同一批原图、相同命名参考照和关系，各自执行元数据与人脸处理。CamRoll 新阶段耗时、资源和失败计入全链路。
2. **共享上游信息的 Agent 对照（可选）**：双方读取固定、相同的预计算元数据/身份结果，比较检索与回答层。必须标明共享组件，并另外报告预处理成本；这不是两套独立全链路比较。

不建议在主实验中让 CamRoll 偷读 Sentrix 已建相册的身份结果：这会引入 Sentrix 处理依赖、缓存收益和潜在后续人工修正。逐图人工 people 标注同样不能默认为允许输入。如果用户明确选择“可信身份标注作为输入”的实验，双方都应获得同样信息并单独命名该条件。

## 8. 优先级与验证

- P0：EXIF完整时间/GPS与来源、地名精度、现有工具过滤和返回字段。这部分输入已经具备，收益依据最直接。
- P1：命名参考登记、人脸检测/匹配、逐脸证据，第六工具及目标人物看图。需要真实组件验证，不能承诺加工具后身份题必对。
- P2：精确聚合查询、匿名聚类、人物画像，按失败案例和实际需要建设。

验证包括：时间与时区缺失、坐标与来源一致性、区县不冒充POI、EXIF旋转后 bbox 对齐、未知/相似人脸、多人同图、错误绑定、跨相册隔离、分页全量计数，以及局部外观属于哪个人。正式100题用于最终评测；阈值与提示词不要按其答案逐题调参。

新的元数据和人脸模块不应改变既有 WeMM encoder/预处理指纹。人脸向量与 WeMM 向量使用不同存储及模型版本。元数据更新发布新快照，人物/地点改变可能使 caption/事件缓存失效；不应因此无条件重算未变化的 WeMM 图片向量。

对前一份接入方案的补充：使用 Qwen参考照对照可以作为候选实验，但不是已验证的人脸聚类/认证实现。若目标是与 Sentrix 的持续人物检索、定位和多图统计能力相匹配，优先建设具有版本、质量门槛和 unknown 状态的人脸匹配组件，工具作为该组件的查询入口。
