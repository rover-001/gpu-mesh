# Real Qwen LoRA training

`train_qwen.py` trains on the selected host's CUDA device, or CPU when CUDA is
unavailable. A llama.cpp inference/RPC server does not perform backpropagation.
The old `best_disaster_adapter.json` contains simulated metadata, not weights;
it is preserved and never accepted as a resumable checkpoint.

Create `.venv-training` with Python 3.11 or newer and install
`scripts/requirements-training.txt`. Install the appropriate PyTorch CPU/CUDA
wheel for the host first. Put the llama.cpp conversion sources matching the
bundled runtime (b12020) in `tools/llama.cpp`, or set `RGPU_LLAMA_CPP_DIR` for the
app. The converter needs `convert_lora_to_gguf.py`, `conversion/`, and `gguf-py/`.
The virtual environment and conversion checkout are ignored by Git.

The trainer uses the original Hugging Face float weights for
`Qwen/Qwen2.5-Coder-0.5B-Instruct`. GGUF is used for inference only. To work
offline, cache that model's config, tokenizer and `model.safetensors` in:

- Linux: `~/.cache/gpu-mesh/training-base/Qwen2.5-Coder-0.5B-Instruct/`
- Windows: `C:/rgpu/training-base/Qwen2.5-Coder-0.5B-Instruct/`

Run from the checkout on Linux:

```sh
.venv-training/bin/python scripts/train_qwen.py --resume --epochs 2 --export-gguf
```

On Windows:

```bat
.venv-training\Scripts\python.exe scripts\train_qwen.py --resume --epochs 2 --export-gguf
```

`--epochs` specifies additional epochs; `--max-steps 1` limits a smoke test.
`--checkpoint <checkpoint-directory>` resumes a specific real checkpoint,
including optimizer and RNG state. Resumed runs default to a lower learning
rate. Model/dataset/split mismatches or corrupt checkpoints fail explicitly.

Checkpoints live in `~/.cache/gpu-mesh/models/checkpoints` on Linux and
`C:/rgpu/models/checkpoints` on Windows. Each checkpoint directory is immutable.
`best.json` advances only when measured heldout loss strictly improves on the
previous best and the initial evaluation. `active_adapter.json` references a
hashed GGUF export; both local inference and remote workers load it with
`--lora` only for the matching Qwen 0.5B base. Restart the inference session
after training. Old model weights and checkpoints remain intact.

For **Peer**, configure `training-peer.json` in the requesting checkout:

```json
{
  "ssh_target": "user@peer-address",
  "root": "/home/user/Projects/rgpu",
  "platform": "linux"
}
```

The remote host needs its own environment, model cache, converter and SSH key
access. Windows peers use `"platform": "windows"` and a root such as `C:/rgpu`.
Optional `python` overrides the remote interpreter. Environment alternatives
are `RGPU_TRAIN_PEER`, `RGPU_TRAIN_PEER_ROOT`, `RGPU_TRAIN_PEER_OS` and
`RGPU_TRAIN_PEER_PYTHON`. Missing configuration reports an error; Peer never
silently trains locally. Training artifacts stay on the training host; copy
the referenced immutable directory and activation manifest together when
deploying to another inference host.

Verification results are in `artifacts/training-validation/`. The current
20-example corpus and 0.5B model still fail answer-quality checks despite
improved heldout loss. Successful training/export is not evidence of accurate
emergency advice. The Windows report records actual base/adapter answers.
