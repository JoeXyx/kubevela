// Package training defines and validates framework-independent SFT configuration
// and generates configuration for a limited LLaMA-Factory runtime profile.
// It does not execute training.
package training

const ConfigVersion = "v1alpha1"

// Config is a proposed configuration contract, not an existing HTTP API.
// Required fields and constraints are documented in README.md.
type Config struct {
	Version    string       `json:"version" yaml:"version"`
	Model      ModelRef     `json:"model" yaml:"model"`
	Dataset    DatasetRef   `json:"dataset" yaml:"dataset"`
	Validation *DatasetRef  `json:"validation,omitempty" yaml:"validation,omitempty"`
	Parameters Parameters   `json:"parameters" yaml:"parameters"`
	Output     OutputConfig `json:"output" yaml:"output"`
}

// ModelRef identifies weights and their associated tokenizer/configuration.
// URI is a reference, not proof of existence or accessibility.
type ModelRef struct {
	URI      string `json:"uri" yaml:"uri"`
	Revision string `json:"revision,omitempty" yaml:"revision,omitempty"`
	// Template is an identifier resolved by a future model adapter, not executable content.
	Template string `json:"template,omitempty" yaml:"template,omitempty"`
}

type DatasetFormat string

const (
	FormatAlpaca   DatasetFormat = "alpaca"
	FormatMessages DatasetFormat = "messages"
)

// DatasetRef describes the platform sample format, not a framework-specific file.
type DatasetRef struct {
	URI     string        `json:"uri" yaml:"uri"`
	Version string        `json:"version,omitempty" yaml:"version,omitempty"`
	Format  DatasetFormat `json:"format" yaml:"format"`
}

type Method string

const (
	MethodFull Method = "full"
	MethodLoRA Method = "lora"
)

type Precision string

const (
	PrecisionFP32 Precision = "fp32"
	PrecisionFP16 Precision = "fp16"
	PrecisionBF16 Precision = "bf16"
)

type Scheduler string

const (
	SchedulerLinear   Scheduler = "linear"
	SchedulerCosine   Scheduler = "cosine"
	SchedulerConstant Scheduler = "constant"
)

type LossScope string

const (
	// LossResponse trains on output tokens for alpaca, or assistant tokens for messages.
	LossResponse LossScope = "response"
	LossAll      LossScope = "all"
)

// Parameters contains training choices, not Kubernetes resource settings.
// Optional pointers distinguish omission from zero/false. No defaults are applied.
type Parameters struct {
	Method                    Method      `json:"method" yaml:"method"`
	Epochs                    float64     `json:"epochs" yaml:"epochs"`
	LearningRate              float64     `json:"learningRate" yaml:"learningRate"`
	BatchSize                 int         `json:"batchSize" yaml:"batchSize"`
	GradientAccumulationSteps int         `json:"gradientAccumulationSteps" yaml:"gradientAccumulationSteps"`
	MaxSequenceLength         int         `json:"maxSequenceLength" yaml:"maxSequenceLength"`
	Precision                 Precision   `json:"precision" yaml:"precision"`
	LossScope                 LossScope   `json:"lossScope" yaml:"lossScope"`
	Scheduler                 Scheduler   `json:"scheduler,omitempty" yaml:"scheduler,omitempty"`
	WarmupRatio               *float64    `json:"warmupRatio,omitempty" yaml:"warmupRatio,omitempty"`
	Seed                      *int64      `json:"seed,omitempty" yaml:"seed,omitempty"`
	GradientCheckpointing     *bool       `json:"gradientCheckpointing,omitempty" yaml:"gradientCheckpointing,omitempty"`
	LoRA                      *LoRAConfig `json:"lora,omitempty" yaml:"lora,omitempty"`
}

// LoRAConfig is required for method=lora and must be absent for method=full.
// TargetModules are explicit model module names, not framework wildcards.
type LoRAConfig struct {
	Rank          int      `json:"rank" yaml:"rank"`
	Alpha         float64  `json:"alpha" yaml:"alpha"`
	Dropout       *float64 `json:"dropout,omitempty" yaml:"dropout,omitempty"`
	TargetModules []string `json:"targetModules" yaml:"targetModules"`
}

// OutputConfig describes artifacts and checkpoint handling.
// URI must eventually resolve to an authorized persistent writable location.
type OutputConfig struct {
	URI            string `json:"uri" yaml:"uri"`
	LoggingSteps   *int   `json:"loggingSteps,omitempty" yaml:"loggingSteps,omitempty"`
	SaveSteps      *int   `json:"saveSteps,omitempty" yaml:"saveSteps,omitempty"`
	SaveTotalLimit *int   `json:"saveTotalLimit,omitempty" yaml:"saveTotalLimit,omitempty"`
}
