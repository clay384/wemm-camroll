# WeMM / CamRoll 项目交接文档

**2026-09-16 当前边界：不修改 `.100`、`.153` 的任何文件；本次只整理本地文档，不连接服务器或执行接口验证。** 此要求覆盖此前实施授权。搭建、改动、源码和接口精确位置见 [CamRoll 搭建与改动总结](CAMROLL_BUILD_SUMMARY_20260916.md)，其中服务器状态均为 2026-09-15 历史观测。

整理日期：2026-09-15。下表主体是先前仅文档交接阶段的 WeMM 历史记录，验收来自 **2026-09-14**。随后用户已授权 CamRoll 改造、独立部署和 D1 API 扩展；新实现入口见 [CamRoll README](../camroll/README.md)、[实现映射](../camroll/docs/IMPLEMENTATION.md)、[部署与恢复](../camroll/docs/DEPLOYMENT.md)。新实验与旧冻结记录分开保存。

| 文档 | 内容 |
|---|---|
| [CamRoll 搭建与改动总结](CAMROLL_BUILD_SUMMARY_20260916.md) | 本轮完整交接：路径、接口、配置、改动、结果、费用与当前只写本地文档边界 |
| [CamRoll 本轮验收](../camroll/docs/VALIDATION_20260915.md) | Qwen/WeMM替换、独立部署、175项组件验证、800题对照与费用 |
| [架构与目录](ARCHITECTURE.md) | 两台服务器职责、模型、数据流、文件与数据库结构 |
| [接口](API.md) | 全部业务 API、参数、返回值、限制及副作用 |
| [调用与运维](USAGE.md) | 登录、检索、上传、导入、复跑及恢复命令 |
| [测试与结果](TESTS_AND_RESULTS.md) | 两轮实验、指标口径、验收证据、未覆盖范围 |
| [交接清单](HANDOFF.md) | 精确路径、当前基线、已知问题和后续工作 |
| [工作约定](../AGENTS.md) | SSH 与历史实施约定；当前操作边界以本页最新说明为准 |

## 快速定位

- 网页地址（历史部署地址）：http://192.168.0.100:8765/
- 服务器项目：`/home/realmagic/wemm-embedding-lab`
- Mac 项目：[wemm-embedding-lab](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab)
- 旧 WeMM 冻结实验：[PhotoBench 报告](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/photobench/20260914-115425/report.html)
- 历史报告：[demo 验收](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/validation_report.html)

以下旧 WeMM 项目实现的是媒体向量检索，返回相似媒体及排名；不生成问题答案，也没有通用身份识别、元数据查询或 router 服务。正式模型名是 `tencent/WeMM-Embedding-9B`。

PhotoBench 图片主测中，原问题 Hit@1/5/10 为 **80.25% / 95.63% / 97.63%**。固定语义改写和关键词方案没有带来总体图片召回收益。所有准确性结论**仅衡量已标注证据**，详见测试文档。
