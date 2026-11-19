# 本周内容

# SFT 训练配置：第一步数据结构

1. `example.json`：一份完整配置。
2. 本文：字段含义和规则。
3. `types.go`：Go 结构体，入口类型为 `training.Config`。

示例只用于讨论和后续测试：model://、dataset://、artifact:// 及示例版本/模板是占位标识，本项目尚无这些标识的解析器。示例数值不是推荐训练参数，模块名也不保证适用于最终模型。该文件不能直接提交到现有 /applications 接口或训练框架。

## 结构

```text
Config
├── version       配置契约版本
├── model         基础模型、版本、模板
├── dataset       训练数据
├── validation    可选独立验证数据
├── parameters    训练方式及超参数
│   └── lora      仅 LoRA 微调时需要
└── output        产物位置、日志和 checkpoint 保存策略
```

首版表达文本 SFT，不含多模态、量化或分布式配置。full 和 lora 是可以表达的配置，不代表当前已具备其训练能力。

## 字段规则（下一步校验器的实现依据）

“必填”指提交 JSON/YAML 时必须提供。省略可选字段表示未指定，不是零值，也不意味着已有默认策略。下一步由服务端预设解析这些值并返回最终生效配置，不依赖不同框架的隐式默认值。

| 字段 | Go 类型 | 必填 | 含义及规则 |
|---|---|---|---|
| version | string | 是 | 固定 v1alpha1，与已有 AIJob apiVersion 是两件事 |
| model.uri | string | 是 | 非空模型引用；此阶段不确认文件存在 |
| model.revision | string | 否 | 非空固定版本；来源可变时，执行前必须解析并记录固定版本或内容摘要 |
| model.template | string | 否 | 模型兼容的模板标识；省略时执行层必须能明确推导，否则拒绝执行 |
| dataset.uri | string | 是 | 非空训练数据引用 |
| dataset.version | string | 否 | 非空数据版本；执行前固定实际内容版本/摘要 |
| dataset.format | DatasetFormat | 是 | alpaca 或 messages，约定见下文 |
| validation | *DatasetRef | 否 | 独立验证集，提供时适用相同引用规则；评估策略尚未设计 |
| parameters.method | Method | 是 | full 或 lora |
| parameters.epochs | float64 | 是 | 正有限数，允许小数轮数；后续运行预设限制上限 |
| parameters.learningRate | float64 | 是 | 正有限数，不接受 NaN/Inf |
| parameters.batchSize | int | 是 | 正整数，每设备批大小 |
| parameters.gradientAccumulationSteps | int | 是 | 正整数，多少小批次后更新一次参数 |
| parameters.maxSequenceLength | int | 是 | 正整数；模型上下文和显存限制在执行前检查 |
| parameters.precision | Precision | 是 | fp32、fp16、bf16；硬件支持另行检查 |
| parameters.lossScope | LossScope | 是 | response 仅回答；all 对有效序列 token 计算损失 |
| parameters.scheduler | Scheduler | 否 | 非空时为 linear、cosine、constant |
| parameters.warmupRatio | *float64 | 否 | 有限数且 0 <= 值 < 1；0 为无预热；constant 不接受正预热比例 |
| parameters.seed | *int64 | 否 | 非负整数；0 是有效值，省略与 0 不同 |
| parameters.gradientCheckpointing | *bool | 否 | true/false，省略与显式 false 不同 |
| parameters.lora | *LoRAConfig | 条件必填 | method=lora 必须提供；method=full 不得提供 |
| parameters.lora.rank | int | 条件必填 | 正整数 |
| parameters.lora.alpha | float64 | 条件必填 | 正有限数 |
| parameters.lora.dropout | *float64 | 否 | 有限数且 0 <= 值 < 1 |
| parameters.lora.targetModules | []string | 条件必填 | 非空列表，元素非空无重复；模块存在性由模型适配层检查 |
| output.uri | string | 是 | 非空产物位置引用，执行前必须授权、持久可写，不直接作为任意文件路径 |
| output.loggingSteps | *int | 否 | 正整数 |
| output.saveSteps | *int | 否 | 正整数 |
| output.saveTotalLimit | *int | 否 | 正整数 |

所有非空字符串拒绝纯空白值。下一步严格解码时拒绝未知字段，避免拼错参数后静默忽略。必填数值省略后为零，应被校验拒绝；可选数值/布尔用指针保存“未指定”和“显式零/false”的区别。首版不使用显式 null 表达业务含义；可选字段缺失或 null 都按未指定处理。

这些是平台拟定规则，不是声称所有训练框架都有相同限制。配置通过语法检查也不代表模型、数据和运行时已经兼容。

## 数据格式约定

平台格式需要后续转换为具体训练框架的数据格式。

### alpaca

每条样本包含非空字符串 instruction 和 output；input 可省略或为空字符串。

```json
{"instruction":"概括这段话","input":"需要概括的正文","output":"期望回答"}
```

lossScope=response 时，仅 output 对应的目标 token 计算损失。

### messages

每条样本包含非空 messages 数组；消息包含 role 和 content，首版角色为 system/user/assistant，content 为非空字符串。至少包含一条 assistant 回答；对话顺序和模板兼容性由后续数据处理校验。

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，有什么可以帮你？"}]}
```

lossScope=response 时，仅 assistant token 计算损失。适配器必须验证模板、截断和训练标签满足这个约定，不得静默改为 all。

## 与当前代码的对应

| 当前 create.js 字段 | 新结构 | 说明 |
|---|---|---|
| modelURI | model.uri | 不代表已经加载模型 |
| datasetURI | dataset.uri | 新增 format，不能仅凭 URI 猜格式 |
| epochs | parameters.epochs | 现前端仅接受正整数；新契约允许正小数，接入时需统一 |
| learningRate | parameters.learningRate | 字符串转换成数值并校验 |
| output.uri（AIJob properties） | output.uri | 后续需要解析成实际产物位置 |
| image、command、CPU/GPU 等 | 不进入本配置 | 属于执行环境，后续通过受控运行配置和资源请求提供 |

本配置不接受客户端自行指定租户身份。真正接入 API 时，身份从登录会话取得；模型/数据/产物授权由服务端执行。

现有页面和接口暂不改动，也不会自动读取这个新结构。下一步实现 Validate(Config)、严格解码和本地参数测试，再接入配置预览。不需要 GPU 或集群就能完成这些工作。




myquestion

微调包括训练这部分是交给KubeVela  来做吗，那我前面是不是就只是相当于一个