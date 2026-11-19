package training

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

func ptr[T any](v T) *T { return &v }

func smokeConfig(t *testing.T) Config {
	t.Helper()
	b, err := os.ReadFile("examples/qwen25-smoke/config.json")
	if err != nil {
		t.Fatal(err)
	}
	c, err := DecodeJSON(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name, field string
		change      func(*Config)
	}{
		{"version", "version", func(c *Config) { c.Version = "v2" }},
		{"model", "model.uri", func(c *Config) { c.Model.URI = "  " }},
		{"revision", "model.revision", func(c *Config) { c.Model.Revision = " " }},
		{"template", "model.template", func(c *Config) { c.Model.Template = " " }},
		{"dataset", "dataset.uri", func(c *Config) { c.Dataset.URI = "" }},
		{"dataset version", "dataset.version", func(c *Config) { c.Dataset.Version = " " }},
		{"format", "dataset.format", func(c *Config) { c.Dataset.Format = "csv" }},
		{"validation", "validation.uri", func(c *Config) { c.Validation = &DatasetRef{Format: FormatAlpaca} }},
		{"method", "parameters.method", func(c *Config) { c.Parameters.Method = "qlora" }},
		{"epochs", "parameters.epochs", func(c *Config) { c.Parameters.Epochs = 0 }},
		{"nan", "parameters.epochs", func(c *Config) { c.Parameters.Epochs = math.NaN() }},
		{"infinity", "parameters.learningRate", func(c *Config) { c.Parameters.LearningRate = math.Inf(1) }},
		{"learning rate", "parameters.learningRate", func(c *Config) { c.Parameters.LearningRate = -1 }},
		{"batch", "parameters.batchSize", func(c *Config) { c.Parameters.BatchSize = 0 }},
		{"accumulation", "parameters.gradientAccumulationSteps", func(c *Config) { c.Parameters.GradientAccumulationSteps = 0 }},
		{"length", "parameters.maxSequenceLength", func(c *Config) { c.Parameters.MaxSequenceLength = -1 }},
		{"precision", "parameters.precision", func(c *Config) { c.Parameters.Precision = "auto" }},
		{"loss", "parameters.lossScope", func(c *Config) { c.Parameters.LossScope = "last" }},
		{"scheduler", "parameters.scheduler", func(c *Config) { c.Parameters.Scheduler = "unknown" }},
		{"warmup upper", "parameters.warmupRatio", func(c *Config) { c.Parameters.WarmupRatio = ptr(1.0) }},
		{"warmup lower", "parameters.warmupRatio", func(c *Config) { c.Parameters.WarmupRatio = ptr(-0.1) }},
		{"warmup nan", "parameters.warmupRatio", func(c *Config) { c.Parameters.WarmupRatio = ptr(math.NaN()) }},
		{"constant warmup", "parameters.warmupRatio", func(c *Config) { c.Parameters.Scheduler = SchedulerConstant; c.Parameters.WarmupRatio = ptr(0.1) }},
		{"seed", "parameters.seed", func(c *Config) { c.Parameters.Seed = ptr(int64(-1)) }},
		{"missing lora", "parameters.lora", func(c *Config) { c.Parameters.LoRA = nil }},
		{"full lora", "parameters.lora", func(c *Config) { c.Parameters.Method = MethodFull }},
		{"rank", "parameters.lora.rank", func(c *Config) { c.Parameters.LoRA.Rank = 0 }},
		{"alpha", "parameters.lora.alpha", func(c *Config) { c.Parameters.LoRA.Alpha = math.Inf(1) }},
		{"dropout", "parameters.lora.dropout", func(c *Config) { c.Parameters.LoRA.Dropout = ptr(1.0) }},
		{"no targets", "parameters.lora.targetModules", func(c *Config) { c.Parameters.LoRA.TargetModules = nil }},
		{"empty target", "parameters.lora.targetModules", func(c *Config) { c.Parameters.LoRA.TargetModules = []string{" "} }},
		{"duplicate target", "parameters.lora.targetModules", func(c *Config) { c.Parameters.LoRA.TargetModules = []string{"q_proj", "q_proj"} }},
		{"output", "output.uri", func(c *Config) { c.Output.URI = "" }},
		{"logging", "output.loggingSteps", func(c *Config) { c.Output.LoggingSteps = ptr(0) }},
		{"save", "output.saveSteps", func(c *Config) { c.Output.SaveSteps = ptr(-1) }},
		{"retention", "output.saveTotalLimit", func(c *Config) { c.Output.SaveTotalLimit = ptr(0) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := smokeConfig(t)
			tt.change(&c)
			if err := Validate(c); err == nil || !strings.Contains(err.Error(), tt.field) {
				t.Fatalf("wanted error for %s, got %v", tt.field, err)
			}
		})
	}
	c := smokeConfig(t)
	c.Parameters.Method, c.Parameters.LoRA = MethodFull, nil
	c.Parameters.Epochs = 0.5
	c.Dataset.Format = FormatMessages
	if err := Validate(c); err != nil {
		t.Fatal(err)
	}
}

func TestDecodeJSON(t *testing.T) {
	c := smokeConfig(t)
	b, _ := json.Marshal(c)
	for name, input := range map[string]string{
		"unknown root":       strings.Replace(string(b), `"version":`, `"typo":1,"version":`, 1),
		"unknown nested":     strings.Replace(string(b), `"batchSize":`, `"batchSzie":`, 1),
		"second object":      string(b) + ` {}`,
		"trailing junk":      string(b) + ` !`,
		"null":               `null`,
		"missing fields":     `{}`,
		"fractional integer": strings.Replace(string(b), `"batchSize":1`, `"batchSize":1.5`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeJSON(strings.NewReader(input)); err == nil {
				t.Fatal("expected rejection")
			}
		})
	}
	c.Parameters.Seed, c.Parameters.GradientCheckpointing = ptr(int64(0)), ptr(false)
	c.Parameters.WarmupRatio = ptr(0.0)
	b, _ = json.Marshal(c)
	decoded, err := DecodeJSON(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Parameters.Seed == nil || *decoded.Parameters.Seed != 0 || decoded.Parameters.GradientCheckpointing == nil || *decoded.Parameters.GradientCheckpointing || decoded.Parameters.WarmupRatio == nil {
		t.Fatal("explicit zero/false was lost")
	}
	c.Parameters.Seed, c.Parameters.GradientCheckpointing, c.Parameters.WarmupRatio = nil, nil, nil
	b, _ = json.Marshal(c)
	decoded, err = DecodeJSON(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Parameters.Seed != nil || decoded.Parameters.GradientCheckpointing != nil || decoded.Parameters.WarmupRatio != nil {
		t.Fatal("decoder applied defaults")
	}
}
