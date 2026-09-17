# CamRoll 相册问答页面独立部署

此部署只管理 `/home/realmagic/camroll-wemm-chat` 中的 8773 进程。它不启停、重启或修改 WeMM 8765、CamRoll 8770、PhotoBench 8772 的源码、配置和进程，也不写入 `.153`。8773 只读访问已冻结的 355 图相册和索引；问答会通过回环地址调用 `127.0.0.1:8772/camroll/api/ask`，实际执行 Agent、WeMM 查询和 Qwen3.7-Flash 云调用，并由 8772 写入原累计预算账本。8773 不持有云密钥、不创建第二套账本、不调用 Judge。

页面按需求不设置登录。同源和 Host 校验用于限制浏览器请求来源，不是身份认证；任何能访问 `192.168.0.100:8773` 的局域网客户端都可能发起有费用的问答。每题可能包含多次 Qwen 调用，费用由 8772 的共享 300 元累计账本预留和结算，不确定调用保留预留。页面不自动重试，全局只允许一个在途问题，其他提交返回 429。

## 固定目录

- release：`/home/realmagic/camroll-wemm-chat/releases/<40位gitcommit>`
- Python：`/home/realmagic/camroll-wemm-chat/.venv/bin/python`
- 运行记录与日志：`/home/realmagic/camroll-wemm-chat/runtime/`，目录权限 0700，文件权限 0600
- 进程入口：`deployment/run_chat.py`
- 应用工厂：`camroll/camroll-chat/camroll_chat/app.py`
- 端口：`0.0.0.0:8773`

release 必须来自已经测试和提交的 Git bundle／干净检出，`HEAD` 必须等于 40 位目录名。发布完成后整棵 release（包括 `.git`）去除写权限。管理器启动前核验 Git commit/tree、干净工作树、无符号链接、关键入口和只读权限；进程记录保存 commit 与 tree。它拒绝源码工作目录、简称提交号、archive 解压目录、可变目录和其他 release 布局。虚拟环境与 8772 的 `.venv-photobench` 分开；不要把依赖安装到 WeMM 环境。

## 创建环境与发布

在可信机器从已经推送的提交创建 Git bundle，在 `.100` 从该 bundle 检出完整提交到上述 release 目录，核验 `HEAD` 和干净状态后执行 `chmod -R a-w "$RELEASE"`。检出时显式指定 `main`，避免 bundle 没有远端默认分支标记时留下未检出的工作树。不要用 `git archive`，因为它没有可供管理器验证的 Git 对象。在该 release 根使用 `camroll/camroll-chat/requirements.txt` 安装页面的固定版本依赖；该文件引用同目录的 `requirements-linux-py310.lock` 约束传递依赖。凭据不得写进 release、命令行或日志。相册、索引和累计预算账本仍由现有 8772/CamRoll 运行根持有，8773 不复制或迁移它们。

```bash
git bundle create /tmp/wemm-camroll-<40位提交号>.bundle main
# 将 bundle 传到 .100 的受限临时目录后：
git clone --branch main /path/to/wemm-camroll-<40位提交号>.bundle \
  /home/realmagic/camroll-wemm-chat/releases/<40位提交号>

install -d -m 0700 /home/realmagic/camroll-wemm-chat/runtime
python3.10 --version
python3.10 -m venv /home/realmagic/camroll-wemm-chat/.venv
RELEASE=/home/realmagic/camroll-wemm-chat/releases/<40位提交号>
cd "$RELEASE"
test "$(git rev-parse HEAD)" = "$(basename "$RELEASE")"
test -z "$(git status --porcelain)"
/home/realmagic/camroll-wemm-chat/.venv/bin/python -m pip install \
  -r "$RELEASE/camroll/camroll-chat/requirements.txt"
chmod -R a-w "$RELEASE"
```

进入当前 release 根后运行：

```bash
/home/realmagic/camroll-wemm-chat/.venv/bin/python deployment/manage_chat.py start
/home/realmagic/camroll-wemm-chat/.venv/bin/python deployment/manage_chat.py status
```

`status` 中 `http_alive=true` 只表示 8773 页面可访问。`/api/health` 会核验冻结快照三件套及 8772 返回的相册/M4 可用性，但不会真正调用 WeMM 或 Qwen。相册、WeMM、Qwen 和上游 8772 的完整链路必须用一次获授权的真实问答验收；不能用进程存活代替完整链路健康。

切换到另一个 release 前必须先显式停止当前 release：

先确认页面没有正在执行的问题。浏览器断线后，8773 后台会继续读取已经发出的上游请求直到终态，并在此期间保持全局执行锁；刷新或重新提交会创建新的云端调用。强停浏览器或 8773 不能撤销已经转交给 8772/Qwen 的调用，可能留下不确定费用，也不能回退余额；管理器不会为中断请求自动补跑。

```bash
/home/realmagic/camroll-wemm-chat/.venv/bin/python deployment/manage_chat.py stop
# 从目标 release 执行 start
```

`restart` 只适用于同一 release 的受控重启：

```bash
/home/realmagic/camroll-wemm-chat/.venv/bin/python deployment/manage_chat.py restart
```

## 进程控制边界

管理器只有在 PID、Linux 启动 tick、完整命令行、cwd、release、host 和端口全部与受限记录一致时才发送 SIGTERM，并使用 pidfd 防止身份校验后的 PID 复用。它拒绝未登记的 8773 监听者，不使用 `pkill`、端口批量杀进程或强制 SIGKILL。

进程记录不匹配时先人工检查 `/proc/<pid>`、端口所有者和受限日志；不要删除记录后直接杀进程。日志是 `runtime/chat-service.log`，不得复制其中可能包含的请求内容到仓库。

## 回滚

```bash
/home/realmagic/camroll-wemm-chat/.venv/bin/python deployment/manage_chat.py rollback
```

回滚只停止经过身份核验的 8773 进程，保留 release 和受限日志供审计。它不恢复或删除冻结相册，不回退累计费用，不重启 8765/8770/8772。需要撤销页面版本时，停止当前 release 后从上一个已验证 release 执行 `start`。

## 本地安全测试

以下测试不启动服务、不访问 `.100/.153`、不调用模型：

```bash
PYTHONPATH=camroll/camroll-chat python -m pytest -q \
  camroll/camroll-chat/tests/test_backend.py \
  deployment/test_chat_deployment.py
node --check camroll/camroll-chat/static/app.js
```

测试覆盖固定 M4/Qwen 策略、相册白名单、安全图片读取、快照与媒体哈希、SSE 脱敏、断流锁、Host/同源校验、上游错误，以及 release 白名单、PID/cwd/argv/启动 tick、旧服务端口保护、未登记监听者、pidfd 停止和受限记录权限。生产依赖不包含 pytest；测试应在开发环境执行，或单独安装固定的 `pytest==8.4.2`。
