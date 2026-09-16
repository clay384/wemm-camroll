# 独立 PhotoBench 部署

此目录只管理新 8772 进程。原 WeMM 8765、CamRoll 8770 和 `.153:8771` 不由这里启停。

## 目录与依赖

- release：`/home/realmagic/camroll-wemm/releases/<40位gitcommit>`；源码从已测试、已提交的仓库部署。
- Python：`/home/realmagic/camroll-wemm/.venv-photobench/bin/python`；依赖见根 `requirements-photobench.txt`，使用 `deployment/requirements-linux-py310.lock` 固定本轮实装的传递依赖，与旧 `.venv` 隔离，使用 CPU ONNX Runtime。
- 私有运行根：`/home/realmagic/camroll-wemm/runtime/photobench-v2`（0700）。其中 `secrets.json`、`judge.key`、`access.key` 均为 0600，不进 Git。
- 数据：`source/album3-max` 保存 manifest、355 张原图、8 张身份参考照和 `100qa-full`；`_transfer_manifest.json` 与相邻传输收据保存 SHA256 证明。
- 输入清单：运行根 `album3-max.spec.json` 只包含原图路径与 manifest 身份/关系，不包含 QA/GT 或逐图人工标签；正式 run 应使用独立 album_id。
- 模型目录：运行根 `models/buffalo_l`；外部部署提供固定人脸检测/识别模型文件，代码不自动下载，不进 Git。
- 新结果与相册分别位于 `evaluation/`、`albums/`；原预算始终为 `/home/realmagic/camroll-wemm/runtime/budget.sqlite`。

## 发布后启动

只有代码通过校验、固定 commit 已发布到 release 目录后才能执行：

```bash
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/manage_eval.py start
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/manage_eval.py status
```

服务使用 `python -m camroll_eval serve --host 0.0.0.0 --port 8772`，由 `run_eval.py` 先加载受限配置再启动。`GET /healthz` 只说明进程存活；有认证的 `/api/health` 才提供应用状态，但也不能代替实际模型接入和已建相册验证。网页登录使用独立 access key，不在日志或终端回显它。

CLI 通过相同配置入口运行，例如：

```bash
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/with_eval_config.py camroll_eval import-history /PRIVATE/HISTORICAL/RUN
```

## 停止与回滚

```bash
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/manage_eval.py stop
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/manage_eval.py rollback
```

两者都只停止经过 PID、启动 tick、完整命令行和 cwd 校验的自有 8772 进程；不强制 kill、不删除新实验、不回退费用账本、不触碰原 8765/8770。原系统完整恢复材料见根 `RESTORE.md`。

## 本地安全测试

```bash
python -m pytest -q deployment/test_eval_deployment.py
```

测试覆盖错误 release、PID 复用、cwd/命令行冲突、误指原服务端口、预算不可重定向和受限文件权限；不启动真实服务或调用模型。
