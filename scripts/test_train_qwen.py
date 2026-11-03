"""Offline checks for loss masking, heldout isolation and checkpoint preservation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("train_qwen", Path(__file__).with_name("train_qwen.py"))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        tokens = [1]
        for message in messages:
            tokens += [2 if message["role"] == "user" else 3]
            tokens += list(range(10, 10 + len(message["content"])))
            tokens += [4]
        if add_generation_prompt:
            tokens += [3]
        return tokens


class TrainerTests(unittest.TestCase):
    def checkpoint(self, directory):
        path = Path(directory) / "epoch-000001-test"
        path.mkdir()
        artifacts = ("adapter_model.safetensors", "adapter_config.json", "training_state.pt")
        for name in artifacts:
            (path / name).write_bytes(b"local-checkpoint-fixture")
        metadata = dict(format_version=1, training_kind="peft_lora_backprop",
                        validation_loss=1.9, base_model=trainer.BASE_MODEL,
                        dataset_sha256="test-dataset",
                        artifact_sha256={name: trainer.file_sha256(path / name) for name in artifacts})
        trainer.atomic_json(path / "metadata.json", metadata)
        return path, metadata

    def test_heldout_is_deterministic_disjoint_and_nonempty(self):
        training, validation = trainer.split_indices(20, 42)
        self.assertEqual((training, validation), trainer.split_indices(20, 42))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(20)))
        self.assertEqual(len(validation), 4)
        self.assertEqual(tuple(map(len, trainer.split_indices(2, 42))), (1, 1))

    def test_response_loss_masks_prompt_and_assistant_header(self):
        messages = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "xy"}]
        encoded = trainer.tokenize_response(FakeTokenizer(), messages, 100)
        self.assertEqual(encoded["labels"], [-100] * 7 + [10, 11, 4])
        self.assertEqual(len(encoded["labels"]), len(encoded["input_ids"]))

    def test_truncated_response_is_rejected_instead_of_training_on_prompt(self):
        messages = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "xy"}]
        with self.assertRaisesRegex(ValueError, "entirely truncated"):
            trainer.tokenize_response(FakeTokenizer(), messages, 7)

    def test_only_strict_measured_improvement_advances_best(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.checkpoint(directory)
            self.assertTrue(trainer.publish_best(directory, path, metadata, 2.0))
            before = (Path(directory) / "best.json").read_bytes()
            self.assertFalse(trainer.publish_best(directory, path, metadata, 1.9))
            self.assertFalse(trainer.publish_best(directory, path, {**metadata, "validation_loss": float("nan")}, 2.0))
            self.assertEqual((Path(directory) / "best.json").read_bytes(), before)
            self.assertEqual(trainer.read_checkpoint(Path(directory) / "best.json")[0], path)

    def test_corrupt_checkpoint_fails_without_cold_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.checkpoint(directory)
            (path / "training_state.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt checkpoint artifact"):
                trainer.read_checkpoint(path)

    def test_missing_optimizer_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.checkpoint(directory)
            (path / "training_state.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                trainer.read_checkpoint(path)

    def test_legacy_simulation_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "best_disaster_adapter.json"
            legacy.write_text('{"best_loss": 0.42}', encoding="utf-8")
            before = legacy.read_bytes()
            with self.assertRaisesRegex(ValueError, "Legacy/simulated"):
                trainer.read_checkpoint(legacy)
            self.assertEqual(legacy.read_bytes(), before)

    def test_checkpoint_pointer_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "best.json"
            pointer.write_text(json.dumps({"checkpoint": "../other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid best checkpoint pointer"):
                trainer.read_checkpoint(pointer)

    def test_output_lock_rejects_concurrent_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with trainer.output_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another trainer"):
                    with trainer.output_lock(directory):
                        self.fail("Concurrent writer acquired lock")
            with trainer.output_lock(directory):
                pass

    def test_export_reuses_immutable_gguf_and_records_real_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.checkpoint(directory)
            adapter = path / "adapter.gguf"
            adapter.write_bytes(b"GGUF-test-fixture")
            before = adapter.stat().st_mtime_ns
            trainer.export_adapter(directory, path, metadata, "converter-not-needed")
            manifest = json.loads((Path(directory) / "active_adapter.json").read_text())
            self.assertEqual(adapter.stat().st_mtime_ns, before)
            self.assertEqual(manifest["adapter_sha256"], trainer.file_sha256(adapter))
            self.assertEqual(manifest["adapter_file"], path.name + "/adapter.gguf")

    def test_gguf_mapping_is_explicit_and_limited(self):
        self.assertEqual(trainer.resolve_model("C:/rgpu/models/" + trainer.BASE_GGUF), trainer.BASE_MODEL)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            trainer.resolve_model("unrelated.gguf")


if __name__ == "__main__":
    unittest.main()
