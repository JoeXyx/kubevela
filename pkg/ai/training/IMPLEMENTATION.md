# SFT 小规模基线与参数生成

更新：2026-11-19，Asia/Shanghai。原有未跟踪的 README.md 保留原样；本文记录本次已实现和已实测的部分。

## 本次结果

已在 `kubevela-sft-lab` 的 RTX 4050 Laptop 6GB 上，用 Go 生成的 YAML 完成真实 SFT、保存 LoRA adapter、重新加载并生成文本。没有接入现有网页或 HTTP API，也没有提交集群任务。

| 项目 | 已验证基线 |
|---|---|
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，约 0.49B 参数 |
| 模型固定 revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| 模板 | `qwen` |
| 数据 | 本目录 examples/qwen25-smoke/alpaca.json，16 条自建短样本 |
| 数据 SHA-256 | `6bfa7fe848fa6fa69735988c9960c444a5448d6c533eaa1c67a792981ee6b421` |
| 训练方式 | LoRA，rank=8，alpha=16，dropout=0，q_proj/v_proj |
| 精度 | BF16，无量化 |
| batch / 梯度累积 | 1 / 4，单卡有效 batch=4 |
| 长度 | cutoff_len=256，本次样本实际最长只有 56 token |
| 轮数 / 更新步数 | 1 / 4 |
| 学习率 | 0.0001，linear，预热比例 0 |
| loss | response，仅回答及回答结束符参与损失 |
| 梯度检查点 | 模型层开启；Trainer 层开关关闭，见兼容说明 |
| seed | 42 |
| checkpoint | 每 4 步保存，保留 1 个；最终 adapter 另外保存 |

最终 Go 配置运行结果：CLI 退出码 0；含加载的 CLI 用时 24.58 秒；训练阶段 5.7581 秒；平均 train_loss=2.0259449034929276；4 次参数更新。每秒采样的整卡显存峰值为 3246MiB，包含桌面、WebUI 等其他进程，不是训练进程独占峰值。

验证脚本检查了全部 16 条样本的有效标签恰为回答及结束符；adapter 所有张量均有限，48 个 LoRA B 张量非零；重新加载基础模型和 adapter 后生成了非空文本。这证明训练和保存/加载链路可用。没有验证集或效果对比，不能据此声称微调提升了模型质量，也不能将短样本显存测量外推到满 256 token 或更大模型。

## 固定运行环境及兼容说明

镜像：`hiyouga/llamafactory@sha256:4811191aff1d59f868416cc9d6f9309c9a6c5fa67f6ce5f4b71783ab142b6b81`。

容器实际版本：LLaMA-Factory `0.9.6.dev0`、Transformers `5.17.0`、PyTorch `2.6.0+cu124`。GPU CUDA 和 BF16 均已实际检查。模型已下载到现有 Hugging Face 缓存卷，后续实验以 `HF_HUB_OFFLINE=1` 运行。

本镜像与常见示例有两处差异，适配器按容器源码和实际执行结果实现：

1. Transformers 5.17.0 不再接受 `warmup_ratio`；`warmup_steps` 小于 1 时表示总步数比例。因此平台 `warmupRatio` 转成 `warmup_steps`，0 仍表示不预热。
2. `gradient_checkpointing=true` 会使 Trainer 向 LLaMA-Factory 的 patched 方法传入不支持的 `every_n_layers`，导致训练失败。实际模型开关是 `disable_gradient_checkpointing`。适配器始终生成 `gradient_checkpointing: false`，并将平台布尔值取反映射到 `disable_gradient_checkpointing`。平台 true 确实启用模型检查点，false 确实关闭；不是忽略这个参数。

开始实验时分别遇到以上两个错误，保留了失败日志；随后一次手工配置训练成功，最后又执行了一次 Go 生成配置并完成全部验证。没有修改容器 Python 源码或升级依赖。更换镜像后应重新验证这些映射。

## Go 接口与范围

```go
config, err := training.DecodeJSON(reader)
// 拒绝未知字段、尾随 JSON、类型错误，并调用 Validate。
// 直接构造 Go Config 时可调用 training.Validate(config)。

bundle, err := training.GenerateLLaMAFactory(config, training.LLaMAFactoryPaths{
    DatasetDir: "/app/data/run-id",
    TrainFile:  "alpaca.json",
    OutputDir:  "/app/saves/run-id",
})
// bundle.TrainYAML 和 bundle.DatasetInfoJSON 是待保存的文件内容。
```

`Validate` 检查必填值、枚举、正数/有限数、比例边界、LoRA 条件字段和重复模块等；可选数值/布尔的省略、0、false 保持区别。它不解析模型引用、不读取数据、不检查硬件。JSON 解码使用 Go 标准 encoding/json；当前没有额外拒绝重复键或大小写别名的策略，接公网 API 前应补齐请求体大小限制与解码策略。

首个转换器只接受上述固定模型 revision、qwen 模板、Alpaca、LoRA、q_proj/v_proj，不接受验证集。通用契约仍能表达 full/messages，但转换器会明确拒绝这些尚未适配的组合。LoRA alpha 的 Go 类型保留 float64，转换前要求正整数且不超过 int32 上限，绝不静默截断。种子限制在运行时支持的 uint32 范围内。

省略可选参数时，本适配器明确使用：scheduler=linear、warmup=0、seed=42、模型梯度检查点=true、dropout=0、loggingSteps=1、saveSteps=100、saveTotalLimit=1。packing=false、mask_history=false、不开评估，不向外部指标服务报告；JSON 解码和 Validate 本身不会补默认值。样例显式指定 saveSteps=4。

`LLaMAFactoryPaths` 是服务端资源准备阶段的输入，不应直接暴露为网页可填字段。调用方必须将数据/产物引用授权并解析成受控容器路径，校验并准备数据、记录摘要、分配新的持久输出目录。这里的路径检查只验证形式，不能代替授权或符号链接检查。`overwrite_output_dir=false` 也不能代替运行前检查：框架可能自动恢复旧 checkpoint；实验脚本会拒绝任何已有输出目录。

## 文件与已保存证据

- `validate.go` / `validate_test.go`：契约校验、JSON 解码及边界测试。
- `llamafactory.go` / `llamafactory_test.go`：YAML/数据登记生成；与实际运行参考文件逐字段比较；不改写输入。
- `examples/qwen25-smoke/config.json`：平台训练配置。
- `examples/qwen25-smoke/train.yaml` / `dataset_info.json`：最终基线参考配置。
- `examples/qwen25-smoke/main.go`：本地生成示例，不启动训练。
- `examples/qwen25-smoke/run_smoke.py`：容器实验脚本，运行 CLI、保存日志、检查标签、重载 adapter；训练最多等待 600 秒。

宿主机已备份：

- `D:\hengxiang_documents\sft-smoke-20261119-generated`：Go 生成的原始配置包。
- `D:\hengxiang_documents\sft-smoke-20261119-result\evidence\verification.json`：最终实际测量和验证结果。
- 同目录 `train.log`、`train.yaml`、`dataset_info.json`、`alpaca.json`：本次执行证据。
- `D:\hengxiang_documents\sft-smoke-20261119-result\adapter`：最终 adapter、tokenizer、指标和 checkpoint-4。

容器内最终产物：`/app/saves/kubevela-sft-smoke-20261119-02`。证据：`/app/data/kubevela-sft-smoke-20261119/runs/kubevela-sft-smoke-20261119-02`。adapter 仍需配合固定版本基础模型使用。

## 复现一个新的实验

下面的目录必须尚不存在；重复运行时换新的 run 名。现有模型缓存可复用。命令在仓库根目录用 PowerShell 执行，每步成功后再执行下一步。

```powershell
go test ./pkg/ai/training/...

go run ./pkg/ai/training/examples/qwen25-smoke `
  -out D:\hengxiang_documents\sft-smoke-next `
  -dataset-dir /app/data/kubevela-sft-smoke-next `
  -output-dir /app/saves/kubevela-sft-smoke-next

docker exec kubevela-sft-lab mkdir /app/data/kubevela-sft-smoke-next
docker cp D:\hengxiang_documents\sft-smoke-next/. kubevela-sft-lab:/app/data/kubevela-sft-smoke-next/
docker cp pkg/ai/training/examples/qwen25-smoke/run_smoke.py kubevela-sft-lab:/app/data/kubevela-sft-smoke-next/run_smoke.py

docker exec -e HF_HUB_OFFLINE=1 kubevela-sft-lab python `
  /app/data/kubevela-sft-smoke-next/run_smoke.py `
  /app/data/kubevela-sft-smoke-next/train.yaml
```

下一步只需接一个配置预览入口：网页提交 Config → 后端解码校验 → 受控资源解析 → 返回生成配置和生效参数。真实任务执行还需资源交付、运行目录隔离及状态/指标采集，本次未将实验脚本当作生产执行器接入。
