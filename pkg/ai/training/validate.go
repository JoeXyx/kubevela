package training

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// DecodeJSON rejects unknown fields and trailing JSON, then validates the contract.
// It does not resolve references, check hardware, or inspect dataset contents.
func DecodeJSON(r io.Reader) (Config, error) {
	var c Config
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("decode training config: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return Config{}, errors.New("training config must contain exactly one JSON object")
	}
	if err := Validate(c); err != nil {
		return Config{}, err
	}
	return c, nil
}

// Validate checks the framework-independent contract without changing c.
func Validate(c Config) error {
	var errs []error
	check := func(ok bool, field, rule string) {
		if !ok {
			errs = append(errs, fmt.Errorf("%s: %s", field, rule))
		}
	}
	text := func(s, field string) { check(strings.TrimSpace(s) != "", field, "must not be empty") }
	optionalText := func(s, field string) {
		if s != "" {
			text(s, field)
		}
	}
	positive := func(n float64, field string) { check(finite(n) && n > 0, field, "must be positive and finite") }
	ratio := func(n *float64, field string) {
		if n != nil {
			check(finite(*n) && *n >= 0 && *n < 1, field, "must be finite and in [0, 1)")
		}
	}
	dataset := func(d DatasetRef, field string) {
		text(d.URI, field+".uri")
		optionalText(d.Version, field+".version")
		check(d.Format == FormatAlpaca || d.Format == FormatMessages, field+".format", "must be alpaca or messages")
	}
	check(c.Version == ConfigVersion, "version", "must be "+ConfigVersion)
	text(c.Model.URI, "model.uri")
	optionalText(c.Model.Revision, "model.revision")
	optionalText(c.Model.Template, "model.template")
	dataset(c.Dataset, "dataset")
	if c.Validation != nil {
		dataset(*c.Validation, "validation")
	}
	p := c.Parameters
	check(p.Method == MethodFull || p.Method == MethodLoRA, "parameters.method", "must be full or lora")
	positive(p.Epochs, "parameters.epochs")
	positive(p.LearningRate, "parameters.learningRate")
	check(p.BatchSize > 0, "parameters.batchSize", "must be positive")
	check(p.GradientAccumulationSteps > 0, "parameters.gradientAccumulationSteps", "must be positive")
	check(p.MaxSequenceLength > 0, "parameters.maxSequenceLength", "must be positive")
	check(p.Precision == PrecisionFP32 || p.Precision == PrecisionFP16 || p.Precision == PrecisionBF16, "parameters.precision", "must be fp32, fp16 or bf16")
	check(p.LossScope == LossResponse || p.LossScope == LossAll, "parameters.lossScope", "must be response or all")
	check(p.Scheduler == "" || p.Scheduler == SchedulerLinear || p.Scheduler == SchedulerCosine || p.Scheduler == SchedulerConstant, "parameters.scheduler", "must be linear, cosine or constant")
	ratio(p.WarmupRatio, "parameters.warmupRatio")
	if p.Scheduler == SchedulerConstant && p.WarmupRatio != nil {
		check(*p.WarmupRatio == 0, "parameters.warmupRatio", "constant scheduler requires zero warmup")
	}
	if p.Seed != nil {
		check(*p.Seed >= 0, "parameters.seed", "must not be negative")
	}
	if p.Method == MethodLoRA {
		check(p.LoRA != nil, "parameters.lora", "is required for lora")
	}
	if p.Method == MethodFull {
		check(p.LoRA == nil, "parameters.lora", "must be absent for full")
	}
	if l := p.LoRA; l != nil {
		check(l.Rank > 0, "parameters.lora.rank", "must be positive")
		positive(l.Alpha, "parameters.lora.alpha")
		ratio(l.Dropout, "parameters.lora.dropout")
		check(len(l.TargetModules) > 0, "parameters.lora.targetModules", "must not be empty")
		seen := map[string]bool{}
		for i, module := range l.TargetModules {
			field := fmt.Sprintf("parameters.lora.targetModules[%d]", i)
			text(module, field)
			check(!seen[module], field, "must not be duplicated")
			seen[module] = true
		}
	}
	text(c.Output.URI, "output.uri")
	for _, entry := range []struct {
		field string
		value *int
	}{
		{"output.loggingSteps", c.Output.LoggingSteps},
		{"output.saveSteps", c.Output.SaveSteps},
		{"output.saveTotalLimit", c.Output.SaveTotalLimit},
	} {
		if entry.value != nil {
			check(*entry.value > 0, entry.field, "must be positive")
		}
	}
	return errors.Join(errs...)
}

func finite(n float64) bool { return !math.IsNaN(n) && !math.IsInf(n, 0) }
