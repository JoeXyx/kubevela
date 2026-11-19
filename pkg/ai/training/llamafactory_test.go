package training

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func smokePaths() LLaMAFactoryPaths {
	return LLaMAFactoryPaths{
		DatasetDir: "/app/data/kubevela-sft-smoke-20261119",
		TrainFile:  "alpaca.json",
		OutputDir:  "/app/saves/kubevela-sft-smoke-20261119-02",
	}
}

// The reference YAML is the independently executed container smoke configuration.
func TestGenerateMatchesSmokeRun(t *testing.T) {
	c := smokeConfig(t)
	before, _ := json.Marshal(c)
	bundle, err := GenerateLLaMAFactory(c, smokePaths())
	if err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(c)
	if string(before) != string(after) {
		t.Fatal("generator mutated the input")
	}
	for name, generated := range map[string][]byte{
		"train.yaml": bundle.TrainYAML, "dataset_info.json": bundle.DatasetInfoJSON,
	} {
		want, err := os.ReadFile("examples/qwen25-smoke/" + name)
		if err != nil {
			t.Fatal(err)
		}
		var gotValue, wantValue any
		if err := yaml.Unmarshal(generated, &gotValue); err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal(want, &wantValue); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gotValue, wantValue) {
			t.Fatalf("%s differs from executed baseline:\ngot %s\nwant %s", name, generated, want)
		}
	}
	data, err := os.ReadFile("examples/qwen25-smoke/alpaca.json")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("sha256:%x", sha256.Sum256(data)); got != c.Dataset.Version {
		t.Fatalf("dataset digest changed: %s", got)
	}
}

func TestGenerateRejectsUnsupported(t *testing.T) {
	for name, change := range map[string]func(*Config){
		"invalid contract": func(c *Config) { c.Parameters.BatchSize = 0 },
		"unknown model":    func(c *Config) { c.Model.URI = "model://unknown" },
		"mutable revision": func(c *Config) { c.Model.Revision = "main" },
		"template":         func(c *Config) { c.Model.Template = "qwen3" },
		"full":             func(c *Config) { c.Parameters.Method = MethodFull; c.Parameters.LoRA = nil },
		"messages":         func(c *Config) { c.Dataset.Format = FormatMessages },
		"validation":       func(c *Config) { d := c.Dataset; c.Validation = &d },
		"fractional alpha": func(c *Config) { c.Parameters.LoRA.Alpha = 8.5 },
		"overflow alpha":   func(c *Config) { c.Parameters.LoRA.Alpha = float64(math.MaxInt32) + 1 },
		"wildcard module":  func(c *Config) { c.Parameters.LoRA.TargetModules = []string{"all"} },
		"comma module":     func(c *Config) { c.Parameters.LoRA.TargetModules = []string{"q_proj,v_proj"} },
		"context":          func(c *Config) { c.Parameters.MaxSequenceLength = 32769 },
		"runtime seed":     func(c *Config) { c.Parameters.Seed = ptr(int64(math.MaxUint32) + 1) },
	} {
		t.Run(name, func(t *testing.T) {
			c := smokeConfig(t)
			change(&c)
			if _, err := GenerateLLaMAFactory(c, smokePaths()); err == nil {
				t.Fatal("expected explicit rejection")
			}
		})
	}
	for name, change := range map[string]func(*LLaMAFactoryPaths){
		"Windows":                func(p *LLaMAFactoryPaths) { p.DatasetDir = `D:\data` },
		"URI":                    func(p *LLaMAFactoryPaths) { p.OutputDir = "artifact://example" },
		"relative":               func(p *LLaMAFactoryPaths) { p.OutputDir = "saves/run" },
		"root":                   func(p *LLaMAFactoryPaths) { p.OutputDir = "/" },
		"unclean":                func(p *LLaMAFactoryPaths) { p.OutputDir = "/app/saves/../data" },
		"overlap":                func(p *LLaMAFactoryPaths) { p.OutputDir = p.DatasetDir + "/output" },
		"parent":                 func(p *LLaMAFactoryPaths) { p.OutputDir = "/app" },
		"same":                   func(p *LLaMAFactoryPaths) { p.OutputDir = p.DatasetDir },
		"traversal":              func(p *LLaMAFactoryPaths) { p.TrainFile = "../other.json" },
		"absolute file":          func(p *LLaMAFactoryPaths) { p.TrainFile = "/other.json" },
		"wrong format":           func(p *LLaMAFactoryPaths) { p.TrainFile = "other.csv" },
		"registration collision": func(p *LLaMAFactoryPaths) { p.TrainFile = "dataset_info.json" },
	} {
		t.Run(name, func(t *testing.T) {
			p := smokePaths()
			change(&p)
			if _, err := GenerateLLaMAFactory(smokeConfig(t), p); err == nil {
				t.Fatal("expected path rejection")
			}
		})
	}
}

func TestGenerateDefaultsAndExplicitValues(t *testing.T) {
	c := smokeConfig(t)
	c.Parameters.Scheduler = ""
	c.Parameters.Seed, c.Parameters.WarmupRatio, c.Parameters.GradientCheckpointing = nil, nil, nil
	c.Parameters.LoRA.Dropout = nil
	c.Output.LoggingSteps, c.Output.SaveSteps, c.Output.SaveTotalLimit = nil, nil, nil
	for _, precision := range []Precision{PrecisionFP32, PrecisionFP16, PrecisionBF16} {
		c.Parameters.Precision = precision
		bundle, err := GenerateLLaMAFactory(c, smokePaths())
		if err != nil {
			t.Fatal(err)
		}
		var args map[string]any
		if err := yaml.Unmarshal(bundle.TrainYAML, &args); err != nil {
			t.Fatal(err)
		}
		if args["bf16"] != (precision == PrecisionBF16) || args["fp16"] != (precision == PrecisionFP16) {
			t.Fatal("wrong precision mapping")
		}
		if args["seed"] != 42 || args["save_steps"] != 100 || args["gradient_checkpointing"] != false || args["disable_gradient_checkpointing"] != false || args["lr_scheduler_type"] != "linear" {
			t.Fatalf("unexpected defaults: %v", args)
		}
	}
	c.Parameters.Seed, c.Parameters.WarmupRatio = ptr(int64(0)), ptr(0.25)
	c.Parameters.GradientCheckpointing = ptr(false)
	c.Parameters.LossScope = LossAll
	bundle, err := GenerateLLaMAFactory(c, smokePaths())
	if err != nil {
		t.Fatal(err)
	}
	var args map[string]any
	if err := yaml.Unmarshal(bundle.TrainYAML, &args); err != nil {
		t.Fatal(err)
	}
	if args["seed"] != 0 || args["warmup_steps"] != 0.25 || args["train_on_prompt"] != true || args["mask_history"] != false {
		t.Fatalf("lost explicit values: %v", args)
	}
	if args["disable_gradient_checkpointing"] != true || args["gradient_checkpointing"] != false {
		t.Fatal("explicit false must disable model checkpointing without enabling the Trainer switch")
	}
	if _, present := args["warmup_ratio"]; present {
		t.Fatal("unsupported Transformers 5.17 argument")
	}
}
