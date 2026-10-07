# 模型架构、外部文件与重建边界

本版本保留应用、检索、模型适配、模型自定义实现、评测和运维源码。**Git clone 可以阅读完整调用结构，但不包含运行所需的权重、私人相册、QA/GT、原始向量、数据库、凭据或已安装环境。** 本次未启动服务、模型或付费请求。

## WeMM：应用适配与模型内部实现

| 层次 | 仓库入口 | 固定来源 |
|---|---|---|
| CamRoll HTTP 适配 | `camroll/camroll-agent/camroll_agent/wemm.py` | 本项目源码 |
| WeMM API 与编码调用 | `wemm/backend/camroll_internal.py`、`encoder.py`、`worker.py`、`preprocess.py` | 现有部署源码；本版不修改这些文件 |
| 官方推理参考和评测源码 | `wemm/official-locked/` | Tencent/WeMM-Embedding commit `6f4f007b9105fe2d0a3176d719b8799e3bf6974e` |
| 模型自定义 Python、结构配置及模板 | [`wemm/model-code/`](../wemm/model-code/README.md) | tencent/WeMM-Embedding-9B revision `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482` |
| 底层 Transformer / tensor 实现 | 外部 Python 包 | [`wemm/requirements.lock`](../wemm/requirements.lock)；不是本仓库自有源码 |

`encoder.py` 从 `wemm/config.yaml` 所指的外部模型目录加载。新增 `model-code/` 用于架构阅读和版本证明，不修改实际加载路径。14 个原版文件的来源 URL、大小、SHA256 见 [`SOURCE_MANIFEST.json`](../wemm/model-code/SOURCE_MANIFEST.json)，均与原 [`SOURCE_LOCK.json`](../wemm/SOURCE_LOCK.json) 一致。

实际运行还需 `SOURCE_LOCK.json` 列出的完整模型文件，包括 **18,815,757,458 字节**的 `model.safetensors` 和 **19,990,378 字节**的 `tokenizer.json`。它们未读取或复制到本版。权重目录、CPU/GPU 环境及设备映射由目标主机独立准备；模型文件存在不代表环境已验证可运行。

WeMM 模型内核基于 Transformers 的 `Qwen3_5ForConditionalGeneration`，模型自定义源码负责最后有效位置池化及归一化；底层通用库实现仍由锁定依赖提供。原模型许可证和第三方条款已保留在 [`wemm/model-code/LICENSE`](../wemm/model-code/LICENSE)。

## Qwen 多模态 embedding 与 reranker

- 独立服务：[`deployment/qwen_worker.py`](../deployment/qwen_worker.py)，回环接口 `/encode`、`/tokenize`、`/split_text`、`/rerank`，串行加载单个模型。
- 检索客户端：[`qwen_retrieval.py`](../camroll/camroll-agent/camroll_agent/qwen_retrieval.py)。图片、文本、事件及查询与固定实验配置绑定。
- 两个实际模型适配实现位于 [`deployment/vendor/qwen_vl/`](../deployment/vendor/qwen_vl/)，上游 `QwenLM/Qwen3-VL-Embedding` revision `393e2978d27852b0d0230d6994f37f9c15bed73c`，来源锁和许可证均保留。
- `SOURCE_LOCK.json` 记录原始上游 SHA256；embedding 文件有已记录的兼容改动，当前有效 SHA256 应按 [`COMPATIBILITY_PATCH.json`](../deployment/vendor/qwen_vl/COMPATIBILITY_PATCH.json) 校验。该改动移除锁定环境不存在的未使用导入；不能把原始哈希与运行哈希不同误判为未知改动。
- 环境入口为 [`qwen_model_worker.requirements.txt`](../deployment/qwen_model_worker.requirements.txt) 及对应 freeze。当前实现使用外部 Transformers、PyTorch、`qwen-vl-utils`，其底层通用实现不复制进本仓库。

两套模型的 **40 个公开文件**来源、每文件 revision、大小和 SHA256 见 [`QWEN_MODEL_FILES.json`](QWEN_MODEL_FILES.json)。此文件仅保存公开模型元数据，去掉了原下载时间和本地绝对路径，不包含文件内容；两个约 4.255 GB 的权重均不入仓。

当前 worker 还要求外部模型根目录中的 `MODEL_WORKER_DOWNLOAD_MANIFEST.json` **原始字节** SHA256 为：

```text
f6ef1a9b61c70cfe836e836e510ace33f40bc9def99e5e8e4417630e53fc68ba
```

这个部署校验清单属于原模型安装材料，内容会参与启动前严格校验。`QWEN_MODEL_FILES.json` 是公开来源投影，**不是该原文件的替代品**；重新序列化或只复制投影不能通过原 worker 校验。恢复既有版本时需由部署保管方提供已校验的原清单及全部模型文件。迁移到其他主机、目录或新安装清单时，应创建新的配置/模型指纹并重新验证，不应直接关闭校验。本版不修改该门槛或已有 WeKnora 安装。

## 其他运行前提

- 人脸检测与识别使用外部 `buffalo_l` ONNX 模型以及 CPU `insightface`/`onnxruntime`，不将模型文件纳入源码。身份参考图和关系资料仍是私人输入。
- Qwen 云生成和 Doubao 校验/评分通过云 API 访问，不存在可随源码提供的本地云模型权重；需要目标账号可用的精确模型 ID、受限凭据和计费设置。
- Python 应用及评测依赖见根 [`requirements-photobench.txt`](../requirements-photobench.txt)、[`deployment/requirements-linux-py310.lock`](../deployment/requirements-linux-py310.lock)。聊天页面有独立的 `camroll/camroll-chat/requirements-linux-py310.lock`。
- 数据库 schema、快照绑定、预算 journal、未知请求处理、Gate 与监督器代码均在仓库中；现有私人账本和运行数据不在源码发布包。恢复代码不得恢复旧费用余额。

## 仅阅读与离线验证

云端代码阅读不需要模型权重或这些外部部署文件。不要为了查看架构而运行 `start`、正式 suite、下载脚本或付费冒烟。非权重模型副本可在仓库根用以下无网络检查核验；此检查不导入模型代码：

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('wemm/model-code')
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
for item in manifest['files']:
    raw = (root / item['path']).read_bytes()
    assert len(raw) == item['bytes'], item['path']
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
print('Verified', len(manifest['files']), 'public non-weight model files')
PY
```

原 `wemm/scripts/package_delivery.py` 是历史交付工具，其范围包括报告和 benchmark；不得拿它收集这次 GitHub 源码发布。发布应由已审核的 Git 跟踪文件清单生成，并在生成后复查排除项。
