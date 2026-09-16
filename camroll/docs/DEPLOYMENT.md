# CamRoll 部署、续跑与恢复

更新日期：**2026-09-15**。适用于 `.100:/home/realmagic/camroll-wemm` 的 Qwen3.7-Flash + WeMM 分支。

## 1. 当前状态与边界

2026-09-15 本轮实施曾观测到 GPU 1 被外部 Sentrix 进程占用、原 WeMM 服务被外部停止；随后资源持有方已恢复 WeMM，新增文本/图片编码及分块接口已实测通过。该历史冲突通过独立 `enrich` 保留进度，未终止外部进程。当前状态仍应以带时间的运行报告及实时检查为准，不能直接复用历史PID执行操作。

用户已明确授权本轮实施 D1，覆盖早先仅交接、禁止写入 `.100` 的限制；授权限于 CamRoll 与已有 WeMM 的 API 扩展、必要恢复及新实验。GPU 0、Sentrix、CamRoll 以外业务和 `.153` 原始资料仍不在可修改范围内。

- **WeMM**：`/home/realmagic/wemm-embedding-lab`，既有端口 8765；固定物理 GPU 1，UUID `GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0`。保持原 BF16、4096 维和像素预算。
- **CamRoll**：`/home/realmagic/camroll-wemm`，独立 `.venv`、`runtime`、相册 SQLite、向量快照及 `reports`。不在此环境加载 WeMM 模型，不下载或复制 RAG 权重。
- **Qwen**：独立 CamRoll 进程访问指南指定的云端业务空间。不要从禁止外网的 WeMM 服务进程中调用云端模型。
- **报告**：每次实际实验使用新 run_id，不覆盖已有 WeMM demo、冻结 PhotoBench 报告或旧向量。
- **外部 GPU 占用**：不终止占用者、不换到 GPU 0、不抢占启动第二份模型。等资源持有方释放并恢复既有 WeMM 后，续跑已保存阶段。

CamRoll HTTP 进程可以在无 GPU、无已发布相册时运行。`GET /api/health` 的 `status=ready` 仅说明 HTTP 应用可用，`albums=[]` 表示没有可查询相册；此时 `/api/ask` 返回 404。**不能据此声称 M1–M4 真实检索、WeMM 或最终回答已经接通。**

## 2. 部署选择与 D1 API 扩展

| 方案 | 做法 | 本轮状态 / 取舍 |
|---|---|---|
| **D1** | CamRoll 独立运行，扩展既有 WeMM API，复用单 worker | 本轮实施；需要 API 层更新和一次受控恢复/重启 |
| D2 | 只使用既有导入及 `/api/search` 接口，另建数据集 | 备选；无需改 WeMM 源码，但无法直接满足当前客户端的向量/tokenize接口和双路索引行为，需要改适配器；不是配置开关 |
| D3 | 独立 WeMM 服务复用磁盘已有权重，与旧服务互斥运行 | 备选；不能同时加载两份模型，也不能控制其他业务释放 GPU；未实施 |

D1 只向 WeMM 添加 `backend/camroll_internal.py`，并在 `backend/app.py` 静态页挂载前注册：

- `POST /api/internal/embeddings`：文本或 CamRoll `runtime` 内受限图片路径 → 归一化 4096 维向量、fingerprint、耗时。
- `POST /api/internal/tokenize`：用同一 WeMM tokenizer 返回 token 数和不超过 512 token 的文本块；原文拼接必须保持一致。

扩展沿用 WeMM 既有认证中间件，只接受 loopback 客户端，调用已有 `worker.request()`。不修改 encoder、worker、预处理、模型配置或权重。超长查询明确报错，Agent 缩短查询；语料文本预先分块。

### 变更前与恢复后核对

以下命令均在 `.100` 的 CamRoll 项目目录执行。`RUN_ID` 必须是本次选定的新报告目录名，替换示例占位值；不要照用旧实验 ID。

```bash
cd /home/realmagic/camroll-wemm
export RUN_ID=REPLACE_WITH_NEW_RUN_ID
mkdir -p "reports/$RUN_ID"
.venv/bin/python deployment/snapshot_wemm.py "reports/$RUN_ID/wemm_before.json"
```

快照脚本读取已有 WeMM SQLite 和文件，调用认证及 `/api/status`；输出只写本次 CamRoll 报告目录。它要求旧服务可访问。当前服务已停止时，不要为生成快照擅自抢占 GPU：保留本轮已取得的 before 快照，等恢复条件满足后再采集 after。

首次安装扩展：

```bash
.venv/bin/python deployment/install_wemm_api.py "$RUN_ID"
```

安装脚本比对 before 快照中的源码哈希，备份 `app.py` 为 `reports/$RUN_ID/wemm_app.py.before`，检查唯一挂载位置后写入 API 扩展。**不是重复运行脚本**：发现已安装标记或源码变化时会拒绝；应核对已有部署，而不是删除保护标记强行再装。

已有 WeMM 正常就绪、无排队/运行资产、GPU 1 确认由原 WeMM 持有时，才可使用：

```bash
.venv/bin/python deployment/restart_wemm.py "$RUN_ID"
.venv/bin/python deployment/snapshot_wemm.py "reports/$RUN_ID/wemm_after.json"
```

`restart_wemm.py` 仅识别 cwd 为既有 WeMM 目录、入口为 `backend.app:app`、端口为 8765 的进程，发送正常退出信号，等待退出后按原环境恢复。**它不是“服务已消失且 GPU 被别人占用”情况下的冷启动工具，也不应绕过前置检查执行。**不要因为看到 8765 无监听就立即启动模型。

before/after 必须核对：

1. `status.fingerprint`、`units` 每条 fingerprint 和旧向量映射相同。
2. `assets` 的 ID、状态、错误，以及 `vector_hashes` 全部相同。
3. `report_hashes` 全部相同；冻结实验结果不应被新建库覆盖。
4. `source_hashes` 中 encoder、worker、preprocess、config 完全相同。只有预期的 API `app.py` 哈希发生变化。
5. 新服务确认使用同一 GPU UUID，且没有第二份模型；记录观测时间、PID、fingerprint 和本次运行证据。

可对已保存快照执行离线比较，不访问模型：

```bash
.venv/bin/python - "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
p = Path('reports') / sys.argv[1]
a = json.loads((p / 'wemm_before.json').read_text())
b = json.loads((p / 'wemm_after.json').read_text())
assert a['status']['fingerprint'] == b['status']['fingerprint']
for key in ('assets', 'units', 'vector_hashes', 'report_hashes'):
    assert a[key] == b[key], key
for name, digest in a['source_hashes'].items():
    if name != 'backend/app.py':
        assert digest == b['source_hashes'][name], name
print('Frozen assets, vectors, fingerprint and reports unchanged')
PY
```

## 3. 独立环境、受限配置与 HTTP 管理

首次环境安装只安装 Python 客户端依赖，不装 torch、Transformers、FAISS 或原 CamRoll 的旧 RAG 模型：

```bash
cd /home/realmagic/camroll-wemm
python3 -m venv .venv
.venv/bin/pip install -r requirements-wemm.txt
.venv/bin/pip install --no-deps -e camroll-agent
```

真实配置由部署方放在 `runtime/secrets.json`，目录权限 0700、文件权限 0600；不纳入 Git、报告或源码交付。不要执行 `cat`、打印环境变量、把账户密钥放在命令行或粘贴到日志。配置值必须都是字符串：

| 名称 | 内容 |
|---|---|
| `DASHSCOPE_API_KEY` | 已授权阿里云凭据，仅由进程读取 |
| `QWEN_BASE_URL` | 交接指南指定的北京业务空间兼容接口，无公共端点回退 |
| `CAMROLL_BUDGET_DB` | 本轮所有 CLI、建库、问答共享的账本绝对路径 |
| `CAMROLL_BUDGET_CNY` | `300` 或更低；不因重启/续跑清零 |
| `CAMROLL_ALBUM_ROOT` | `/home/realmagic/camroll-wemm/runtime/albums` |
| `WEMM_BASE_URL` | `http://127.0.0.1:8765` |
| `WEMM_KEY_PATH` | `/home/realmagic/wemm-embedding-lab/run/access.key`，只保存文件路径 |

`deployment/with_config.py` 在进程内读取受限配置，再运行模块或脚本；不会打印配置。`deployment/run_service.py` 先加载同一配置，再导入应用。必须从项目根目录运行，确保相对输出目录一致。

### 只管理 CamRoll 自身

```bash
.venv/bin/python deployment/manage_camroll.py start
.venv/bin/python deployment/manage_camroll.py status
.venv/bin/python deployment/manage_camroll.py stop
```

管理器仅启动固定入口 `deployment/run_service.py`，固定绑定 `127.0.0.1:8770`、单 worker。进程记录和日志为 `runtime/camroll-service.json`、`runtime/camroll-service.log`。停止前同时校验 cwd、完整命令行、Linux 进程创建 tick，并使用 pidfd 绑定具体进程实例；身份不一致、PID 被复用或缺少 pidfd 支持时拒绝发信号。它不会控制 WeMM、GPU、其他端口或系统级 systemd。

启动遇到已有未管理的 8770 监听者时拒绝；不要通过杀占用者解决。`status` 仅验证管理器记录的进程和监听状态，不证明模型或索引就绪。`stop` 只发正常退出信号，不发送强制 kill。

本机连通性检查（不调用计费模型）：

```bash
curl --fail http://127.0.0.1:8770/api/health
```

Mac 访问通过 SSH 隧道，不把 8770 绑定到外网：

```bash
ssh -N -L 8770:127.0.0.1:8770 realmagic@192.168.0.100
```

已有已发布相册后才能进行问答接口检查；以下调用会计费：

```bash
curl --fail http://127.0.0.1:8770/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"album_id":"photobench-album3","question":"这些照片记录了什么？","config":"M1"}'
```

`stream=true` 返回 SSE 工具事件，最终 `answer` 事件包含证据 ID、index_version、stop reason 和本题 usage。它不是逐 token 输出。`stopped_reason=ok` 表示正常结束，不等于最终答案正确率已经评测。

## 4. 相册首次处理、续跑与发布

输入仅包含照片及可信元数据；PhotoBench 的 QA 答案、证据标签和人工事件标注不能进入建库内容。完整原图和派生图均位于独立资产目录，不修改源照片。相同内容按 SHA256 去重，来源路径和日期独立保存。

以下示例使用已生成的 `runtime/photobench-album3.json`。普通相册也可使用 `images: [{"path":"绝对路径","date":"YYYY-MM-DD"}]`；日期可缺失。先检查清单：

```bash
.venv/bin/python deployment/with_config.py camroll_agent inspect runtime/photobench-album3.json
```

### GPU 不可用时：enrich

```bash
.venv/bin/python deployment/with_config.py camroll_agent enrich \
  runtime/photobench-album3.json -o runtime/albums/photobench-album3 \
  --concurrency 4 --resume
```

`enrich` 仅执行预处理、Qwen 中文 caption 和事件整理，不创建 WeMM 客户端，不发布向量索引。写入阶段缓存、`enriched_records.json` 和 `last_enrich.json`；已有 `CURRENT` 保持不变。首次只有 enrich 完成时，服务仍不能查询该相册。

`embed-images SPEC -o MEMORY --resume` 只顺序预建图片向量缓存，可与同一清单的 `enrich` 并行；不调用Qwen、不改事件或CURRENT。两阶段结束后运行 `build --resume` 汇合、补文本编码并发布。指纹变化时停止发布并保留缓存，不混合不同模型的向量。

小样本接入可追加 `--max-images 4`；确认实际 usage 和费用估算后再去掉限制续跑全量。小样本成功阶段会复用；事件的上下文变化可能使受影响事件重新生成，不能把整库费用简单当成零。

### WeMM 恢复后：首次 build 或正常续跑

```bash
.venv/bin/python deployment/with_config.py camroll_agent build \
  runtime/photobench-album3.json -o runtime/albums/photobench-album3 \
  --concurrency 4 --resume
```

`build` 与 `run` 等价：在同一目录复用成功 caption/事件与编码阶段，只补失败、缺失或版本失效项；`--resume` 默认开启。显式 `--force` 会跳过阶段缓存并重新调用云端，不适合作为普通故障恢复手段。

有可用图片向量时，构建写入新的 `snapshots/<version>`，校验后通过原子替换 `CURRENT` 发布；单次查询始终使用一个不可变版本。发布可能是部分覆盖，必须检查 `image_count`、`image_embedding_count`、`caption_count`、`text_coverage`、`errors` 和 `published`。`published=true` 不保证 334 张均成功；不得把部分索引当成完整对照。

### 已有已发布索引后：index

```bash
.venv/bin/python deployment/with_config.py camroll_agent index runtime/albums/photobench-album3
```

`index` 读取已有 `CURRENT` 的照片/事件记录，复用有效缓存并补建 WeMM 图片/文本向量，**不调用云端生成**。当前实现要求已发布索引：只有 enrich、尚无 CURRENT 的首次部署应执行 `build --resume`，不能直接执行 `index`。

预算耗尽或已归类的建库/索引提供方失败时，CLI 输出可恢复 summary，退出码为 3。读取对应 `last_enrich.json`、`last_build.json`、`last_index.json` 以及阶段缓存确定进度；先修复依赖或确认预算，再执行同一命令。模型尚未就绪的前置连接错误可能发生在 summary 产生之前，不能据此认为已有缓存被删除。

```bash
.venv/bin/python deployment/with_config.py camroll_agent budget
```

保留同一 `CAMROLL_BUDGET_DB`，不要通过换路径、删除账本或重建 300 元额度继续超预算任务。费用预留、不确定请求和实际 usage 含义见 [Qwen 运行说明](QWEN_RUNTIME.md)。

## 5. 回滚、验证与已知限制

### CamRoll 发布回滚

1. 保留失败版本及日志作为证据，不删除 assets、阶段缓存或预算账本。
2. 选择已有、已验证的 `snapshots/<version>`；检查该版本 `records.json`、向量文件和 metadata 中的 WeMM fingerprint。
3. 先用临时文件写入该版本 ID，再用 `os.replace` 原子替换该相册 `CURRENT`。服务后续请求加载这个版本；正在进行的请求继续使用自己已固定的旧实例。
4. `GET /api/health` 只能检查相册是否发布。恢复后的 M1–M4 问答或检索应分别取得新证据，不能复用旧“健康”声明。

### WeMM API 层回滚

如果 before/after 的旧 fingerprint、资产状态、旧向量或报告哈希出现非预期变化，先暂停新 CamRoll 编码和评测，保留证据。不要触发旧库重建、导入、清理或 recover 来掩盖差异。

只有确认现有 `backend/app.py` 仍是本轮安装的版本且没有他人后续改动时，才能从本次 `wemm_app.py.before` 恢复 API 入口；同时移除本轮新增注册/扩展文件，保留原 encoder、worker、preprocess、config 与权重。若源码已被他人修改，先合并核对，禁止直接覆盖。

恢复 API 后仍须遵守 GPU 归属和 WeMM 恢复条件：仅在 GPU 1 可由既有服务使用时恢复，重新采集 after 并逐项比较。绝不终止外部 Sentrix PID、改用 GPU 0 或加载第二份 WeMM。

### 本地验证

```bash
PYTHONPATH=camroll-agent .venv/bin/python -m pytest -q tests camroll-agent/tests
```

关键覆盖包括坏 JSON/参数纠错、工具错误与零结果区分、view 次数、最终图片/引用传递、同步与 SSE 等价、跨线程实际请求预算、ASGI 断连清理、相册路径越界、费用并发预留、阶段续跑、稳定事件 ID 和检索融合。

目前无账号系统、无前端、无批量最终答案评分。M1/M3 的检索实测和 M2/M4 的最终图片传递必须在依赖恢复后取得真实报告；HTTP 进程启动和模拟测试通过不能替代这些验收。
