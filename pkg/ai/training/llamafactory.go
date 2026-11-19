package training

import (
	"encoding/json"
	"fmt"
	"math"
	"path"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	Qwen25SmallModel    = "Qwen/Qwen2.5-0.5B-Instruct"
	Qwen25SmallRevision = "7ae557604adf67be50417f59c2c2f167def9a775"
)

// LLaMAFactoryPaths must come from a trusted resource resolver, not HTTP input.
// The caller must authorize Dataset.URI/Output.URI, stage and validate the data,
// record its digest, and reserve a new persistent output directory before execution.
// These are Linux container paths, never host Windows paths or platform URIs.
type LLaMAFactoryPaths struct {
	DatasetDir string
	TrainFile  string // basename within DatasetDir; JSON or JSONL Alpaca records
	OutputDir  string
}

// LLaMAFactoryBundle contains files to stage, not a runnable platform Job.
type LLaMAFactoryBundle struct {
	TrainYAML       []byte
	DatasetInfoJSON []byte
}

// GenerateLLaMAFactory implements the first, deliberately narrow adapter:
// pinned Qwen2.5-0.5B-Instruct, qwen template, Alpaca data, q_proj/v_proj LoRA,
// single-device text SFT without evaluation. Targets LLaMA-Factory 0.9.6.dev0
// with Transformers 5.17.0 (warmup_steps < 1 is a ratio in this version).
// Unsupported choices fail explicitly.
// It does not download, authorize, write files, or start training.
func GenerateLLaMAFactory(c Config, paths LLaMAFactoryPaths) (LLaMAFactoryBundle, error) {
	fail := func(message string) (LLaMAFactoryBundle, error) {
		return LLaMAFactoryBundle{}, fmt.Errorf("llamafactory: %s", message)
	}
	if err := Validate(c); err != nil {
		return LLaMAFactoryBundle{}, err
	}
	if c.Model.URI != Qwen25SmallModel || c.Model.Revision != Qwen25SmallRevision || c.Model.Template != "qwen" {
		return fail("first adapter requires the pinned Qwen2.5-0.5B-Instruct model and qwen template")
	}
	if c.Parameters.Method != MethodLoRA {
		return fail("first adapter supports only lora")
	}
	if c.Dataset.Format != FormatAlpaca || c.Validation != nil {
		return fail("first adapter requires alpaca training data without a validation set")
	}
	p, l := c.Parameters, c.Parameters.LoRA
	// Alpha remains float64 in the framework-independent contract. Never truncate it.
	if math.Trunc(l.Alpha) != l.Alpha || l.Alpha > math.MaxInt32 {
		return fail("parameters.lora.alpha must be an integer in [1, 2147483647]")
	}
	if p.MaxSequenceLength > 32768 {
		return fail("parameters.maxSequenceLength exceeds this model's 32768-token context")
	}
	for _, module := range l.TargetModules {
		if module != "q_proj" && module != "v_proj" {
			return fail("first adapter supports only q_proj and v_proj target modules")
		}
	}
	if p.Seed != nil && *p.Seed > math.MaxUint32 {
		return fail("parameters.seed must fit the runtime's unsigned 32-bit seed")
	}
	for _, dir := range []string{paths.DatasetDir, paths.OutputDir} {
		if !path.IsAbs(dir) || path.Clean(dir) != dir || dir == "/" || strings.ContainsAny(dir, "\\\x00\r\n") {
			return fail("dataset/output directories must be clean absolute Linux container paths")
		}
	}
	if paths.OutputDir == paths.DatasetDir || strings.HasPrefix(paths.OutputDir, paths.DatasetDir+"/") || strings.HasPrefix(paths.DatasetDir, paths.OutputDir+"/") {
		return fail("dataset/output directories must be separate, non-overlapping locations")
	}
	if paths.TrainFile == "" || path.Base(paths.TrainFile) != paths.TrainFile || strings.ContainsAny(paths.TrainFile, "\\\x00\r\n") || (path.Ext(paths.TrainFile) != ".json" && path.Ext(paths.TrainFile) != ".jsonl") || paths.TrainFile == "dataset_info.json" {
		return fail("train file must be a JSON/JSONL basename distinct from dataset_info.json")
	}
	scheduler := p.Scheduler
	if scheduler == "" {
		scheduler = SchedulerLinear
	}
	seed := valueOr(p.Seed, int64(42))
	args := map[string]any{
		"model_name_or_path": c.Model.URI, "model_revision": c.Model.Revision,
		"trust_remote_code": false, "stage": "sft", "do_train": true,
		"finetuning_type": "lora", "lora_rank": l.Rank, "lora_alpha": int64(l.Alpha),
		"lora_dropout": valueOr(l.Dropout, 0.0), "lora_target": strings.Join(l.TargetModules, ","),
		"template": c.Model.Template, "dataset": "sft_train", "dataset_dir": paths.DatasetDir,
		"cutoff_len": p.MaxSequenceLength, "train_on_prompt": p.LossScope == LossAll,
		"mask_history": false, "packing": false,
		"preprocessing_num_workers": 1, "dataloader_num_workers": 0,
		"output_dir": paths.OutputDir, "overwrite_output_dir": false,
		"logging_steps": valueOr(c.Output.LoggingSteps, 1), "save_strategy": "steps",
		"save_steps": valueOr(c.Output.SaveSteps, 100), "save_total_limit": valueOr(c.Output.SaveTotalLimit, 1),
		"report_to": "none", "per_device_train_batch_size": p.BatchSize,
		"gradient_accumulation_steps": p.GradientAccumulationSteps,
		"learning_rate":               p.LearningRate, "num_train_epochs": p.Epochs,
		"lr_scheduler_type": string(scheduler), "warmup_steps": valueOr(p.WarmupRatio, 0.0),
		"bf16": p.Precision == PrecisionBF16, "fp16": p.Precision == PrecisionFP16,
		// Use LLaMA-Factory's model switch. Enabling the Trainer switch also
		// calls the patched method with unsupported every_n_layers in 5.17.0.
		"gradient_checkpointing":         false,
		"disable_gradient_checkpointing": !valueOr(p.GradientCheckpointing, true),
		"seed":                           seed, "data_seed": seed, "do_eval": false, "eval_strategy": "no", "val_size": 0,
	}
	train, err := yaml.Marshal(args)
	if err != nil {
		return LLaMAFactoryBundle{}, fmt.Errorf("marshal training YAML: %w", err)
	}
	info, err := json.MarshalIndent(map[string]any{
		"sft_train": map[string]any{
			"file_name": paths.TrainFile, "formatting": "alpaca",
			"columns": map[string]string{"prompt": "instruction", "query": "input", "response": "output"},
		},
	}, "", "  ")
	if err != nil {
		return LLaMAFactoryBundle{}, fmt.Errorf("marshal dataset registration: %w", err)
	}
	return LLaMAFactoryBundle{TrainYAML: train, DatasetInfoJSON: append(info, '\n')}, nil
}

func valueOr[T any](value *T, fallback T) T {
	if value != nil {
		return *value
	}
	return fallback
}
