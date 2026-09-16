# WeMM + CamRoll 私有源码备份

这是 2026-09-16 从 `.100` 实际部署只读采集的源码基线，不包含模型权重、私人相册、QA、向量、数据库、运行结果或凭据。当前结构为 CamRoll 五工具 Agent、Qwen3.7-Flash 云端生成和既有 WeMM 单 GPU worker。

- 基线标签：`baseline/2026-09-16-pre-photobench`。
- 源码采集时间及每个来源文件的 SHA256：[`baseline/source_manifest.json`](baseline/source_manifest.json)。
- `camroll/`：`/home/realmagic/camroll-wemm` 当前源码、测试、部署工具和项目文档。
- `wemm/`：`/home/realmagic/wemm-embedding-lab` 当前源码、前端、固定配置及官方来源锁；包含已部署的 `backend/camroll_internal.py`。
- `docs/project/`：Mac 本地交接及后续设计文档的原样存档；其中旧路径和旧授权边界是历史记录，实际操作遵循最新用户授权与根目录约定。
- `baseline/environment/`：采集时的 Python 版本与 `pip freeze`，不包含虚拟环境本体。CamRoll 的 editable 路径仅记录原部署位置，重建时使用本地 `camroll-agent` 路径。

## 架构

原图与可信日期进入 CamRoll 的可恢复建库流程，Qwen 生成描述和事件，WeMM 编码图片与文本，生成独立相册快照。Agent 通过 `search`、`grep`、`list_by_date`、`get`、`view_image` 读取同一个固定索引版本，返回答案、证据及工具轨迹。WeMM 复用既有 worker，不启动第二份模型。

本基线尚未实现拟议的人脸身份工具、完整时间/GPS 扩展、多轮 Benchmark 协议或独立评测页面。后续开发应在基线标签之后单独提交，评测使用新相册和新 run_id。

## 重建与恢复

恢复说明见 [`RESTORE.md`](RESTORE.md)。源码基线、运行数据备份和凭据是三个独立部分；Git clone 本身不能恢复相册数据库或预算账本。本次只进行了源码采集与备份，没有执行恢复、停止服务或调用计费模型。

当前真实部署的 WeMM API 已有后续认证/调度保护，本仓库保存的是采集时版本。不要拿历史报告中的 `wemm_app.py.before` 代替这一基线；那是早于 CamRoll 接入的版本。

`wemm/scripts/package_delivery.py` 是原部署中保留的历史源码，其旧打包白名单会包含运行报告，不得用于 GitHub 备份。向本仓库提交时遵守根 `.gitignore` 和明确的源码范围。

## 来源与授权

CamRoll 基于 `thaoshibe/camroll` 的 `1967f977dbd51ce3f3b3b0d3213d21a514e55dfd`，保留其 CC BY-NC-SA 4.0 许可证及修改记录。WeMM 官方代码和模型版本见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 与 `wemm/SOURCE_LOCK.json`。本仓库不把第三方源码重新声明为自有许可。
