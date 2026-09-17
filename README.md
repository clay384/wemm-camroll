# WeMM + CamRoll 与独立 PhotoBench 评测

仓库保存 2026-09-16 从 `.100` 实际部署采集的完整基线，以及基于该基线开发的六工具 CamRoll 与独立 PhotoBench 评测。生成使用 Qwen3.7-Flash，向量复用既有 WeMM 单 GPU worker。仓库不包含模型权重、私人相册、QA、向量、数据库、逐题运行结果或凭据。

- 基线标签：`baseline/2026-09-16-pre-photobench`。
- 源码采集时间及每个来源文件的 SHA256：[`baseline/source_manifest.json`](baseline/source_manifest.json)。
- `camroll/`：`/home/realmagic/camroll-wemm` 当前源码、测试、部署工具和项目文档。
- `camroll/camroll-chat/`：固定 355 图相册的浅色问答页面、同源代理、安全图片读取和实时可审计轨迹。
- `deployment/CHAT.md`：独立 8773 进程的发布、验证和回滚说明。
- `wemm/`：`/home/realmagic/wemm-embedding-lab` 当前源码、前端、固定配置及官方来源锁；包含已部署的 `backend/camroll_internal.py`。
- `docs/project/`：Mac 本地交接及后续设计文档的原样存档；其中旧路径和旧授权边界是历史记录，实际操作遵循最新用户授权与根目录约定。
- `baseline/environment/`：采集时的 Python 版本与 `pip freeze`，不包含虚拟环境本体。CamRoll 的 editable 路径仅记录原部署位置，重建时使用本地 `camroll-agent` 路径。

## 架构

原图 EXIF 时间/GPS、离线地名、命名参考照和关系进入独立建库。CPU 人脸模型提供带来源和不确定状态的身份匹配，Qwen 生成描述和事件，WeMM 编码图片与文本，发布独立相册快照。Agent 通过 `search`、`grep`、`list_by_date`、`get`、`view_image`、`query_photo_people` 读取固定索引，支持隔离多轮会话并分别记录候选、证据和最终交付图片。详细契约见 [`docs/PHOTOBENCH_PROTOCOL.md`](docs/PHOTOBENCH_PROTOCOL.md)。

`camroll/camroll-agent/camroll_eval/` 提供独立 runner、冻结的 Sentrix 评分兼容层、共享费用账本上的 Judge 调用、历史回放和局域网直接访问的比较页面。正式协议为 355 图、8 人、6 条关系、100 题／111 轮，Agent/Judge 并发 12/12。原 `.153:8771` 保持只读；新服务使用 `.100:8772`，旧 `.100:8765/8770` 保留。

相册问答数据流为 `浏览器 → 8773 → 8772 CamRoll Agent → Qwen3.7-Flash / WeMM → 冻结索引`。8773 固定相册、索引版本、M4 和模型，不持有云密钥或第二套费用账本；提问仍会产生云调用并写入 8772 的共享账本。8773 的回滚只停止自己的受控进程，不能撤销上游调用或恢复费用余额。

## 重建与恢复

恢复说明见 [`RESTORE.md`](RESTORE.md)。源码基线、运行数据备份和凭据是三个独立部分；Git clone 本身不能恢复相册数据库。累计费用账本始终沿用，不能恢复旧余额。`deployment/manage_eval.py` 只管理经进程身份核验的独立 8772 服务，`rollback` 停止该服务，原 8770 继续可用。

部署目录固定为 `/home/realmagic/camroll-wemm/releases/<完整提交号>`，使用独立 `.venv-photobench`。受限配置位于 `runtime/photobench-v2`。部署依赖锁见 `deployment/requirements-linux-py310.lock`；`deployment/with_eval_config.py camroll_eval run --run-id <新ID>` 启动正式冷建库评测，完成后的 run 不可改写。

当前真实部署的 WeMM API 已有后续认证/调度保护，本仓库保存的是采集时版本。不要拿历史报告中的 `wemm_app.py.before` 代替这一基线；那是早于 CamRoll 接入的版本。

`wemm/scripts/package_delivery.py` 是原部署中保留的历史源码，其旧打包白名单会包含运行报告，不得用于 GitHub 备份。向本仓库提交时遵守根 `.gitignore` 和明确的源码范围。

## 来源与授权

CamRoll 基于 `thaoshibe/camroll` 的 `1967f977dbd51ce3f3b3b0d3213d21a514e55dfd`，保留其 CC BY-NC-SA 4.0 许可证及修改记录。WeMM 官方代码和模型版本见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 与 `wemm/SOURCE_LOCK.json`。本仓库不把第三方源码重新声明为自有许可。
