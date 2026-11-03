#!/usr/bin/env python3
"""Real response-only Qwen LoRA training with measured heldout checkpoint gating.

Dependencies: pip install -r scripts/requirements-training.txt
Only use checkpoints produced locally: training_state.pt contains trusted Python state.
The mesh-server option is retained for CLI compatibility; this process trains on
its own CUDA device (or CPU). A llama.cpp inference server cannot backpropagate.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
BASE_GGUF = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
FORMAT_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(value):
    if value.lower().endswith(".gguf"):
        if value.replace("\\", "/").split("/")[-1].lower() != BASE_GGUF:
            raise ValueError("GGUF is inference-only; supply a Hugging Face model directory/ID for training")
        print(f"[MODEL] GGUF is inference-only; loading original float weights: {BASE_MODEL}", flush=True)
        return BASE_MODEL
    return value


def load_dataset(path):
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        if messages is None and "instruction" in record:
            messages = [
                {"role": "user", "content": (record["instruction"] + "\n" + record.get("input", "")).strip()},
                {"role": "assistant", "content": record.get("output", "")},
            ]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Invalid messages on dataset line {line_number}")
        if any(m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"Invalid message role/content on dataset line {line_number}")
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise ValueError(f"Dataset line {line_number} needs a final assistant response")
        # One training example per conversation; earlier assistant turns are context.
        samples.append(messages)
    if len(samples) < 2:
        raise ValueError("At least two conversations are required for a disjoint heldout split")
    return samples


def split_indices(count, seed, fraction=0.2):
    if count < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two samples and a validation fraction between 0 and 1")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    heldout_count = max(1, min(count - 1, math.ceil(count * fraction)))
    return indices[heldout_count:], indices[:heldout_count]


def tokenize_response(tokenizer, messages, max_length):
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template prompt is not a token prefix of the response; cannot mask safely")
    ids = full[:max_length]
    labels = [-100] * min(len(prompt), len(ids)) + ids[len(prompt):]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("Response was entirely truncated; increase --max-length")
    return {"input_ids": ids, "labels": labels}


def read_checkpoint(path):
    path = Path(path).resolve()
    if path.is_file():
        if path.name != "best.json":
            raise ValueError(f"Legacy/simulated JSON is not a trainable checkpoint; preserved unchanged: {path}")
        pointer = json.loads(path.read_text(encoding="utf-8"))
        relative = Path(pointer.get("checkpoint", ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid best checkpoint pointer")
        path = (path.parent / relative).resolve()
    required = ("adapter_model.safetensors", "adapter_config.json", "training_state.pt", "metadata.json")
    for name in required:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: missing {path / name}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("training_kind") != "peft_lora_backprop":
        raise ValueError("Checkpoint is not a real supported LoRA training checkpoint")
    hashes = metadata.get("artifact_sha256", {})
    if set(hashes) != {"adapter_model.safetensors", "adapter_config.json", "training_state.pt"}:
        raise ValueError("Checkpoint integrity hashes are missing")
    for name, expected in hashes.items():
        if name not in required or file_sha256(path / name) != expected:
            raise ValueError(f"Corrupt checkpoint artifact: {name}")
    if not math.isfinite(metadata["validation_loss"]):
        raise ValueError("Checkpoint validation loss is not finite")
    return path, metadata


def publish_best(output, checkpoint, metadata, previous_loss):
    loss = metadata["validation_loss"]
    if not math.isfinite(loss) or not loss < previous_loss:
        return False
    atomic_json(Path(output) / "best.json", {
        "format_version": FORMAT_VERSION,
        "checkpoint": Path(checkpoint).name,
        "validation_loss": loss,
        "base_model": metadata["base_model"],
        "dataset_sha256": metadata["dataset_sha256"],
    })
    return True


def log_telemetry(step, total_steps, epoch, total_epochs, loss, best_loss, lr, tok_sec,
                  messages=None, status="Training"):
    messages = messages or []
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    response = messages[-1]["content"] if messages else ""
    payload = dict(type="train_telemetry", step=step, total_steps=total_steps, epoch=epoch,
                   total_epochs=total_epochs, loss=float(loss), best_loss=float(best_loss),
                   lr=float(lr), tok_sec=float(tok_sec), sample_title=prompt[:70],
                   sample_category="Disaster Management", prompt_preview=prompt[:280],
                   response_preview=response[:340], status=status, timestamp=time.time())
    print("[TELEMETRY] " + json.dumps(payload, allow_nan=False), flush=True)


@contextmanager
def output_lock(output):
    """OS-held lock is released on exit/crash; no stale lock deletion required."""
    with (Path(output) / ".training.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another trainer/exporter owns this checkpoint output directory") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def export_adapter(output, checkpoint, metadata, llama_cpp_dir):
    if metadata["base_model"] != BASE_MODEL:
        raise ValueError("Automatic app activation currently supports only the Qwen 0.5B base")
    destination = Path(checkpoint) / "adapter.gguf"
    if not destination.exists():
        converter = Path(llama_cpp_dir) / "convert_lora_to_gguf.py"
        if not converter.is_file():
            raise ValueError(f"LoRA converter missing: {converter}; set --llama-cpp-dir")
        temporary = destination.with_name(".adapter-" + uuid.uuid4().hex + ".gguf")
        base_path = metadata.get("base_model_path")
        base_args = ["--base", base_path] if base_path and Path(base_path).is_dir() else ["--base-model-id", metadata["base_model"]]
        subprocess.run([sys.executable, str(converter), *base_args,
                        "--outfile", str(temporary), "--outtype", "f16", str(checkpoint)], check=True)
        with temporary.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("Converter did not produce a GGUF adapter")
        os.rename(temporary, destination)
    with destination.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Existing immutable adapter is not a GGUF file")
    atomic_json(Path(output) / "active_adapter.json", {
        "format_version": FORMAT_VERSION, "base_model": BASE_MODEL,
        "base_gguf_filename": BASE_GGUF,
        "adapter_file": f"{Path(checkpoint).name}/adapter.gguf",
        "adapter_sha256": file_sha256(destination),
    })
    print(f"[EXPORT] Active inference adapter: {destination}", flush=True)


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        _train(args)


def _train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = args.checkpoint
    if args.resume and not resume:
        if (output / "best.json").exists():
            resume = str(output / "best.json")
        elif (output / "best_disaster_adapter.json").exists():
            print("[WARNING] Legacy simulated best_disaster_adapter.json has no learned weights; preserving it and loading pretrained HF weights.", flush=True)
    if args.export_only:
        selected, metadata = read_checkpoint(output / "best.json")
        export_adapter(output, selected, metadata, args.llama_cpp_dir)
        return
    checkpoint, previous = read_checkpoint(resume) if resume else (None, None)
    model_id = resolve_model(args.model)
    model_source = args.base_model or model_id
    cached_base = Path("C:/rgpu/training-base/Qwen2.5-Coder-0.5B-Instruct") if os.name == "nt" else Path.home() / ".cache/gpu-mesh/training-base/Qwen2.5-Coder-0.5B-Instruct"
    if not args.base_model and model_id == BASE_MODEL and (cached_base / "config.json").is_file():
        model_source = str(cached_base)
    samples = load_dataset(args.data)
    data_hash = file_sha256(args.data)
    train_ids, validation_ids = split_indices(len(samples), args.seed)
    if previous:
        identity = (model_id, data_hash, args.max_length, args.seed)
        saved_identity = (previous["base_model"], previous["dataset_sha256"], previous["max_length"], previous["seed"])
        if identity != saved_identity or train_ids != previous["train_indices"] or validation_ids != previous["validation_indices"]:
            raise ValueError("Resume model/dataset/tokenization/split does not match checkpoint; refusing cold restart")
    if (output / "best.json").exists():
        _, existing = read_checkpoint(output / "best.json")
        if (existing["base_model"], existing["dataset_sha256"], existing["max_length"], existing["seed"]) != (model_id, data_hash, args.max_length, args.seed):
            raise ValueError("Output already contains a best checkpoint for different training data/configuration")
    else:
        existing = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as error:
        raise RuntimeError("Install real training dependencies: python -m pip install -r scripts/requirements-training.txt") from error
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    revision = previous.get("model_revision") if previous else None
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint) if checkpoint else model_source, revision=revision)
    encoded = [tokenize_response(tokenizer, sample, args.max_length) for sample in samples]
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype, revision=revision).to(device)
    model.config.use_cache = False
    model_revision = getattr(model.config, "_commit_hash", None)
    if checkpoint:
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)
    else:
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    if device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    lr = args.lr if args.lr is not None else (5e-5 if checkpoint else 1e-4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    if checkpoint:
        # Only load trusted local checkpoints; RNG state requires Python deserialization.
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    def batch(index):
        item = encoded[index]
        return {name: torch.tensor([value], dtype=torch.long, device=device) for name, value in item.items()}
    def evaluate():
        model.eval()
        total_loss, tokens = 0.0, 0
        with torch.no_grad():
            for index in validation_ids:
                item = batch(index)
                count = int((item["labels"][:, 1:] != -100).sum().item())
                value = float(model(**item).loss.item())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite heldout loss; refusing checkpoint promotion")
                total_loss += value * count
                tokens += count
        return total_loss / tokens
    baseline = evaluate()
    baseline_limit = min(baseline, previous["baseline_validation_loss"]) if previous else baseline
    best_loss = min(baseline_limit, existing["validation_loss"]) if existing else baseline_limit
    print(f"[VALIDATION] Measured starting heldout loss: {baseline:.6f}; device={device}", flush=True)
    completed = previous["completed_epochs"] if previous else 0
    offset = previous.get("epoch_step", 0) if previous else 0
    global_step = previous["global_step"] if previous else 0
    final_epoch = completed + args.epochs
    planned_steps = args.epochs * len(train_ids) - offset
    if args.max_steps:
        planned_steps = min(planned_steps, args.max_steps)
    total_steps = global_step + planned_steps
    run_steps = 0
    last_loss = baseline
    for epoch in range(completed + 1, final_epoch + 1):
        order = list(train_ids)
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        end_offset = 0
        for position in range(offset, len(order)):
            index = order[position]
            item = batch(index)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            loss = model(**item).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss; checkpoint preserved")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            last_loss = float(loss.item())
            speed = len(encoded[index]["input_ids"]) / max(time.perf_counter() - start, 1e-9)
            global_step += 1
            run_steps += 1
            end_offset = position + 1
            log_telemetry(global_step, total_steps, epoch, final_epoch, last_loss, best_loss, lr, speed, samples[index])
            if args.max_steps and run_steps >= args.max_steps:
                break
        validation_loss = evaluate()
        finished_epoch = end_offset == len(order)
        metadata = dict(format_version=FORMAT_VERSION, training_kind="peft_lora_backprop",
                        base_model=model_id, base_model_path=str(Path(model_source).resolve()) if Path(model_source).is_dir() else None, model_revision=model_revision, dataset_sha256=data_hash,
                        completed_epochs=epoch if finished_epoch else epoch - 1,
                        epoch_step=0 if finished_epoch else end_offset,
                        global_step=global_step, train_indices=train_ids, validation_indices=validation_ids,
                        seed=args.seed, max_length=args.max_length, baseline_validation_loss=baseline_limit, starting_validation_loss=baseline,
                        validation_loss=validation_loss, learning_rate=lr, device=device,
                        saved_at=time.time(), parent_checkpoint=str(checkpoint) if checkpoint else None)
        destination = output / f"epoch-{epoch:06d}-step-{global_step:08d}-{uuid.uuid4().hex[:8]}"
        staging = output / (".pending-" + uuid.uuid4().hex)
        staging.mkdir()
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        torch.save(dict(optimizer=optimizer.state_dict(), python_rng=random.getstate(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None),
                   staging / "training_state.pt")
        metadata["artifact_sha256"] = {name: file_sha256(staging / name) for name in
                                      ("adapter_model.safetensors", "adapter_config.json", "training_state.pt")}
        atomic_json(staging / "metadata.json", metadata)
        os.rename(staging, destination)
        improved = publish_best(output, destination, metadata, best_loss)
        if improved:
            best_loss = validation_loss
            print(f"[CHECKPOINT] Strict heldout improvement; best={destination}", flush=True)
            if args.export_gguf:
                export_adapter(output, destination, metadata, args.llama_cpp_dir)
        else:
            print(f"[PRESERVE] heldout={validation_loss:.6f}; best={best_loss:.6f} unchanged; resumable checkpoint={destination}", flush=True)
        offset = 0
        if args.max_steps and run_steps >= args.max_steps:
            break
    log_telemetry(global_step, total_steps, epoch, final_epoch, validation_loss, best_loss, lr, 0.0, status="Completed")


def parse_args(argv=None):
    root = Path(__file__).resolve().parent.parent
    output = Path("C:/rgpu/models/checkpoints") if os.name == "nt" else Path.home() / ".cache/gpu-mesh/models/checkpoints"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--base-model", default=None, help="Local original HF float weights; preserves --model identity")
    parser.add_argument("--data", default=str(root / "data/disaster_management_train.jsonl"))
    parser.add_argument("--output", default=str(output))
    parser.add_argument("--epochs", type=int, default=2, help="Additional epochs, including the unfinished epoch on resume")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit optimizer steps in this invocation for smoke testing")
    parser.add_argument("--mesh-server", default=None, help="Compatibility only: training runs on this host, not the inference server")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--export-gguf", "--export", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Export existing validated best without training")
    parser.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", str(root / "tools/llama.cpp")))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.lora_r < 1 or args.max_length < 2 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("epochs/rank/max-steps must be positive; max-length must be at least 2")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("learning rate must be finite and positive")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception as error:
        print(f"[ERROR] Real training stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
