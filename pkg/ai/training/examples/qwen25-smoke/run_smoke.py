"""Run inside the pinned lab container: python run_smoke.py /path/to/train.yaml.

Runs a bounded local smoke test, checks labels, and reloads the saved adapter.
Requires the companion 16-record alpaca.json and dataset_info.json in dataset_dir.
Never reuses an output directory. This is a lab script, not a platform executor.
"""

import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

import yaml


config_path = pathlib.Path(sys.argv[1]).resolve()
args = yaml.safe_load(config_path.read_text())
output = pathlib.Path(args["output_dir"])
data_dir = pathlib.Path(args["dataset_dir"])
if output.exists():
    raise SystemExit("Refusing to reuse output directory: " + str(output))
evidence = data_dir / "runs" / output.name
evidence.mkdir(parents=True, exist_ok=False)
for source in (config_path, data_dir / "dataset_info.json", data_dir / "alpaca.json"):
    shutil.copy2(source, evidence / source.name)

start = time.monotonic()
peak = 0
with (evidence / "train.log").open("x") as log:
    proc = subprocess.Popen(
        ["llamafactory-cli", "train", str(config_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    while proc.poll() is None:
        sample = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if sample.returncode == 0:
            peak = max(peak, int(sample.stdout.strip().splitlines()[0]))
        if time.monotonic() - start > 600:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise SystemExit("Training exceeded the 600-second smoke limit")
        time.sleep(1)

result = {
    "exit_code": proc.returncode,
    "cli_elapsed_seconds": round(time.monotonic() - start, 2),
    "peak_device_used_mib_including_other_processes": peak,
    "dataset_sha256": hashlib.sha256((data_dir / "alpaca.json").read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
}
(evidence / "run-result.json").write_text(json.dumps(result, indent=2))
if proc.returncode:
    print((evidence / "train.log").read_text()[-8000:])
    raise SystemExit(proc.returncode)

# Verify every sample through the same LLaMA-Factory preprocessing used by SFT.
import torch
import transformers
import llamafactory
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

model_args, data_args, training_args, _, _ = get_train_args(args)
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)["train_dataset"]
samples = json.loads((data_dir / "alpaca.json").read_text())
assert len(dataset) == len(samples) == 16
for row, sample in zip(dataset, samples):
    labels = row["labels"]
    supervised = [token for token in labels if token != -100]
    assert labels[0] == -100 and supervised
    assert len(row["input_ids"]) <= args["cutoff_len"]
    assert tokenizer.decode(supervised) == sample["output"] + "<|im_end|>\n"

# Saved LoRA B starts at zero by default; nonzero finite B verifies an update.
weights = load_file(str(output / "adapter_model.safetensors"))
assert all(torch.isfinite(t).all().item() for t in weights.values())
nonzero_b = sum(bool(torch.count_nonzero(t).item()) for name, t in weights.items() if "lora_B" in name)
assert nonzero_b > 0
metrics = json.loads((output / "train_results.json").read_text())
state = json.loads((output / "trainer_state.json").read_text())
assert state["global_step"] == 4 and math.isfinite(metrics["train_loss"])

base = AutoModelForCausalLM.from_pretrained(
    args["model_name_or_path"], revision=args["model_revision"],
    dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
).to("cuda")
model = PeftModel.from_pretrained(base, str(output), is_trainable=False).eval()
saved_tokenizer = AutoTokenizer.from_pretrained(str(output), trust_remote_code=False)
text = saved_tokenizer.apply_chat_template(
    [{"role": "user", "content": "Translate to English: apple"}],
    tokenize=False, add_generation_prompt=True,
)
inputs = saved_tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
answer = saved_tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
assert answer.strip()
result.update({
    "versions": {"llamafactory": llamafactory.__version__, "torch": torch.__version__, "transformers": transformers.__version__},
    "model_revision": args["model_revision"],
    "global_step": state["global_step"], "metrics": metrics,
    "samples_with_verified_response_only_labels": len(samples),
    "max_actual_sequence_length": max(len(row["input_ids"]) for row in dataset),
    "nonzero_lora_b_tensors": nonzero_b,
    "adapter_reload": "passed", "reload_generation": answer,
})
(evidence / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
