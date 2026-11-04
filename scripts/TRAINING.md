# Actual disaster-model training and deployment

The active model is **Qwen/Qwen2.5-1.5B-Instruct** with a real PEFT LoRA adapter.
The failed 0.5B coding model is retired from the app's model list. The reference
answer bypass has been removed: all chat responses come from llama-server model
inference. The chat template supplies only general role instructions, never
question matches, reference passages, or predefined answers.

Training runs on the selected host's CUDA device, or CPU when unavailable.
Install `scripts/requirements-training.txt` in `.venv-training`. The converter
source matching the bundled llama.cpp runtime lives at `tools/llama.cpp`.

Offline original HF float weights/config/tokenizer:

- Rover/Rahul: `~/.cache/gpu-mesh/training-base/Qwen2.5-1.5B-Instruct/`
- Windows: `C:/rgpu/training-base/Qwen2.5-1.5B-Instruct/`

Inference base: `qwen2.5-1.5b-instruct-q4_k_m.gguf` from the official Qwen GGUF repo.
Its SHA256 is `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
It must be paired with the 1.5B adapter, never the old 0.5B adapter.

Training uses `data/disaster_management_verified.jsonl`: 120 concise sourced
question/answer variants. Source metadata is for corpus provenance; it is not
retrieved at inference time. `data/disaster_generation_eval.json` contains
separate prompt phrasings for actual generated-answer checks. These small probes
are a regression check, not a comprehensive disaster-response certification.

Continue the deployed best checkpoint on Linux:

```sh
.venv-training/bin/python scripts/train_qwen.py --resume --epochs 2 --export-gguf
```

On Windows, use `.venv-training\Scripts\python.exe` with the same arguments.
Defaults select the verified corpus, 384-token training length and family-specific
output directory. Peer training uses `training-peer.json` with `ssh_target`,
`root`, `platform` and optional `python`; SSH authentication must already work.

Checkpoint locations:

- Linux: `~/.cache/gpu-mesh/models/checkpoints-disaster-1.5b/`
- Windows: `C:/rgpu/models/checkpoints-disaster-1.5b/`

Deploy the immutable epoch directory plus `best.json` and `active_adapter.json`.
The manifest records the matching base identity and exported adapter hash. The
app verifies both before using `--lora`. Copy full optimizer/tokenizer files when
further local training is required. Preserve checkpoints; `best.json` advances
only for strictly improved measured validation loss. `--checkpoint` resumes
weights, optimizer and RNG. `--fine-tune-data --checkpoint <path>` explicitly
continues learned weights on a changed corpus and starts a new epoch cursor;
use a separate output directory because old/new validation losses differ.

Both local and peer chat use port 8080 with transparent byte forwarding. Local
llama-server uses 18081, peer worker uses 18080. There is no answer interception.
Rebuild/restart the app on all devices after changing model selection or adapter
integration; restart inference after changing the deployed adapter.

Actual GPU training and raw llama-server generation reports are saved under
`artifacts/training-validation/`. No Git push is performed for this change.
