# 第三方来源

| 组件 | 来源与固定版本 | 许可与保留材料 |
|---|---|---|
| CamRoll | https://github.com/thaoshibe/camroll — `1967f977dbd51ce3f3b3b0d3213d21a514e55dfd` | `camroll/LICENSE`：CC BY-NC-SA 4.0；`camroll/docs/UPSTREAM_README.md` 保留上游介绍；部署修改见其实现文档 |
| WeMM 官方代码 | https://github.com/Tencent/WeMM-Embedding — `6f4f007b9105fe2d0a3176d719b8799e3bf6974e` | `wemm/official-locked/LICENSE`：Tencent 官方代码 Apache-2.0；保留官方 README 和源码 |
| WeMM 模型 | `tencent/WeMM-Embedding-9B` — `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482` | 保存来源锁及 `wemm/model-code/` 中的非权重架构材料；权重不入仓，使用时另行遵循模型授权 |
| Qwen VL embedding／reranker代码 | https://github.com/QwenLM/Qwen3-VL-Embedding — `393e2978d27852b0d0230d6994f37f9c15bed73c` | `deployment/vendor/qwen_vl/LICENSE`、`SOURCE_LOCK.json`与`COMPATIBILITY_PATCH.json`；权重和tokenizer外置，详见 `docs/MODEL_DEPENDENCIES.md` |

本次备份不复制原 Git 对象历史，而以来源 URL、固定 commit 和逐文件哈希记录出处。新增部署、适配和文档不改变上述原有许可声明。

2026-10-07架构发布补充的WeMM模型定义及非权重配置，其逐文件来源与哈希见 `wemm/model-code/SOURCE_MANIFEST.json` 和 `docs/MODEL_DEPENDENCIES.md`，不包含权重。

历史元数据差异：`camroll/camroll-agent/pyproject.toml` 仍含MIT标签，而 `camroll/LICENSE` 与原交接为CC BY-NC-SA 4.0。本次仅归档并披露差异，不通过README或包元数据重新许可第三方代码。
