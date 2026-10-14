# 云端阅读仓库与访问验证

**2026-10-14观测：** GitHub API显示本仓库目前为PUBLIC，10月7日发布时为PRIVATE；本次未改变可见性。当前入口先读[完整交接与Gate-Off方案](PROJECT_HANDOFF_20261014.md)，再读源码地图。公开源码可以通过GitHub/Raw地址直接读取，但不能据此认定账号其他私有项目已授权。下面的连接流程保留为私有仓库访问说明；仓库公开状态和GitHub connector账号授权是不同事实。

仓库：`clay384/wemm-camroll`。本架构版本标签：`architecture/2026-10-07-v1`。

## 连接与验证

1. 在 ChatGPT 的 Plugins／Apps 中找到 GitHub，使用拥有本仓库读取权限的 GitHub 账号完成连接；仓库选择中允许访问 `clay384/wemm-camroll`。组织策略适用时，还需完成对应批准。
2. 在启用了 GitHub 工具的新对话中明确给出仓库名和版本标签。仅粘贴私有仓库链接、仅在本机登录 GitHub、或仅安装插件，都不能证明云端已取得该仓库读取权限。
3. 先请求读取本版本 `README.md`、本文件和 `docs/ARCHITECTURE_CURRENT.md`；要求回复实际读取的路径及版本，不接受依靠仓库名称推测架构。
4. 如果返回 404／无可访问仓库，检查连接账号、所选仓库范围和组织授权。仓库保持 Private 即可；不需要为了 ChatGPT 阅读而改为 Public。

访问验收分开记录：GitHub 私有发布成功、GitHub 文件下载校验成功、ChatGPT GitHub 工具实际读取成功是三项独立结果。前两项通过不能代替第三项。账户连接状态会变化，本文件不宣称任何当前账号已经连接成功。

官方操作说明：[插件连接](https://learn.chatgpt.com/docs/plugins)、[服务访问权限](https://learn.chatgpt.com/docs/enterprise/apps-and-connectors)、[Codex Cloud](https://learn.chatgpt.com/docs/cloud)。功能是否可用取决于账号和工作区设置。

## 推荐的首次阅读请求

> 使用 GitHub 连接读取私有仓库 clay384/wemm-camroll，固定标签 architecture/2026-10-07-v1。先阅读 README.md、AGENTS.md、docs/ARCHITECTURE_CURRENT.md、docs/CODE_MAP.md、docs/MODEL_DEPENDENCIES.md 和 docs/releases/2026-10-07-architecture-v1.md。列出你实际读取的文件，然后解释建库、WeMM／Qwen 检索、六工具 Agent、Gate、费用与请求 journal、Doubao 最终评分、报告服务和 GPU 恢复之间的关系。区分当前架构、历史实现、外部运行资源和未完成评测。仅做源码阅读，不执行建库、模型请求、部署或正式评测。

后续可按 [源码地图](CODE_MAP.md) 阅读具体模块。不要把整个仓库、全部历史文档和逐次工具结果反复塞进一个上下文；以目录地图定位，再读取相关完整函数和测试。当前 Gate 的上下文保护问题见版本说明，不能自动改规则或补跑正式题。

## 阅读完整架构与实际运行的区别

- GitHub 工具可读取授权源码；若需要克隆、执行测试或修改，使用有对应仓库权限的 Codex Cloud 环境。
- 架构阅读和离线组件测试不需要权重、私人媒体、QA／GT 或云密钥。
- 真实模型运行需要单独准备 [模型依赖](MODEL_DEPENDENCIES.md)、服务、凭据和私人运行数据。它们不在源码包中。
- 仓库内的历史部署脚本含本地服务器路径、固定相册／输入哈希及 GPU1 约束。云端克隆不会自动获得该局域网或相册；这些脚本不是任意环境的一键安装器。
- 当前累计费用账本是独立持久状态。不能为了云端运行创建新空账本绕过累计上限，也不能从历史备份回退余额。
- `8774` 是已有运行记录的只读页面。其源数据未上传，源码包不会包含那些私人逐题答案或图片。
