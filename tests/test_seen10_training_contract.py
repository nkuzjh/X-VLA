import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from csgo_seen10.dataset import resolve_seen10_augmentation
from train_seen10 import (
    _FAIR_POLICY,
    _LEGACY_POLICY,
    _checkpoint_data_cursor,
    _canonical_resume_action_contract,
    _fair_train_loader_seed,
    _keep_frozen_vision_eval,
    _merged_inference_state,
    _parameter_role,
    _prune_periodic_checkpoints,
    _resolve_gradient_accumulation,
    _resolve_train_loader_budget,
    _restore_full_resume_state,
    _save_full_resume_state,
    _phase_metadata,
    _update_seen10_lrs,
    _validate_config,
)


class Seen10TrainingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/csgo_seen10_xvla_fair.json").read_text(encoding="utf-8"))

    def test_fair_config_is_the_fixed_seen10_contract(self):
        contract = _validate_config(self.config)
        self.assertEqual(contract["mode"], "official_auto")
        self.assertEqual(contract["noise_width"], 20)
        self.assertEqual(contract["loss_dimensions"], 5)
        self.assertFalse(contract["dummy_channels_in_loss"])
        self.assertEqual(self.config["training"]["iters"], 19_500)
        self.assertEqual(self.config["training"]["eval_interval"], 3_900)

    def test_augmentation_is_explicit_and_train_only(self):
        self.assertFalse(resolve_seen10_augmentation("seen_train"))
        self.assertTrue(resolve_seen10_augmentation("seen_train", True))
        self.assertFalse(resolve_seen10_augmentation("seen_validation", True))
        self.assertFalse(resolve_seen10_augmentation("seen_discrete_test", True))

    def test_effective_batch_is_128_for_supported_world_sizes(self):
        train_cfg = self.config["training"]
        expected = {1: 32, 2: 16, 4: 8, 8: 4}
        for world_size, accumulation in expected.items():
            with self.subTest(world_size=world_size):
                batching = _resolve_gradient_accumulation(
                    train_cfg,
                    batch_size=4,
                    world_size=world_size,
                    smoke=False,
                )
                self.assertEqual(batching["effective_batch_size"], 128)
                self.assertEqual(batching["gradient_accumulation_steps"], accumulation)
        self.assertEqual(50_000 // 128, 390)
        self.assertEqual(50_000 - 390 * 128, 80)
        self.assertEqual(19_500 // 390, 50)

    def test_checkpoint_epoch_boundary_points_to_next_unread_batch(self):
        self.assertEqual(_checkpoint_data_cursor(9, 12_480, 12_480), (10, 0))
        self.assertEqual(_checkpoint_data_cursor(9, 6_240, 12_480), (9, 6_240))

    def test_fair_and_legacy_loader_budgets_follow_sampler_drop_policies(self):
        fair = _resolve_train_loader_budget(
            50_000,
            batch_size=4,
            world_size=1,
            gradient_accumulation_steps=32,
            policy=_FAIR_POLICY,
        )
        self.assertEqual(fair["sampler_samples_per_rank"], 50_000)
        self.assertEqual(fair["loader_micro_batches_per_epoch"], 12_500)
        self.assertEqual(fair["updates_per_epoch"], 390)
        self.assertEqual(fair["micro_batches_per_epoch"], 12_480)
        self.assertEqual(fair["dropped_samples_per_epoch"], 80)

        legacy = _resolve_train_loader_budget(
            50_000,
            batch_size=4,
            world_size=1,
            gradient_accumulation_steps=1,
            policy=_LEGACY_POLICY,
        )
        self.assertEqual(legacy["sampler_samples_per_rank"], 50_000)
        self.assertEqual(legacy["loader_micro_batches_per_epoch"], 12_500)
        self.assertEqual(legacy["updates_per_epoch"], 12_500)
        self.assertEqual(legacy["micro_batches_per_epoch"], 12_500)

    def test_fair_epoch_rank_seed_is_stable_and_distinct(self):
        self.assertEqual(_fair_train_loader_seed(42, 1, 3), _fair_train_loader_seed(42, 1, 3))
        self.assertNotEqual(_fair_train_loader_seed(42, 0, 3), _fair_train_loader_seed(42, 1, 3))
        self.assertNotEqual(_fair_train_loader_seed(42, 1, 3), _fair_train_loader_seed(42, 1, 4))

    def test_fair_frozen_backbones_eval_but_lora_dropout_and_connectors_train(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm = torch.nn.Module()
                self.vlm.vision_tower = torch.nn.Sequential(torch.nn.Dropout())
                self.vlm.language_model = torch.nn.Sequential(torch.nn.Dropout())
                self.vlm.lora_dropout = torch.nn.Dropout()
                self.transformer = torch.nn.Module()
                self.transformer.blocks = torch.nn.Sequential(torch.nn.Dropout())
                self.connector = torch.nn.Sequential(torch.nn.Dropout())

        model = TinyModel().train()
        _keep_frozen_vision_eval(model, _FAIR_POLICY)
        self.assertFalse(model.vlm.vision_tower.training)
        self.assertFalse(model.vlm.language_model.training)
        self.assertFalse(model.transformer.blocks.training)
        self.assertTrue(model.vlm.lora_dropout.training)
        self.assertTrue(model.connector.training)

    def test_legacy_phase_metadata_and_action_alias_resume_are_canonical(self):
        training = self.config["training"]
        early = _phase_metadata(training, 0, _LEGACY_POLICY)
        late = _phase_metadata(training, 1000, _LEGACY_POLICY)
        self.assertEqual(early["phase"], "legacy_freeze_lr")
        self.assertEqual(early["active_lr_roles"], ["soft_prompts", "action_heads"])
        self.assertEqual(late["phase"], "legacy_native_schedule")
        self.assertEqual(
            late["active_lr_roles"],
            ["vlm", "transformer_core", "soft_prompts", "action_heads"],
        )
        old = {"mode": "auto", "external_action_dim": 5, "model_action_dim": 20}
        current = {"mode": "legacy_reset_dummy", "external_action_dim": 5, "model_action_dim": 20}
        self.assertEqual(_canonical_resume_action_contract(old), current)

    def test_fair_learning_rate_phase_boundary(self):
        parameters = {role: torch.nn.Parameter(torch.ones(())) for role in (
            "action_heads",
            "soft_prompt",
            "llm_lora",
            "action_expert_lora",
            "vision_language_connector",
            "action_connector",
        )}
        optimizer = torch.optim.AdamW([
            {"name": role, "params": [parameter], "lr": 0.0}
            for role, parameter in parameters.items()
        ])
        _update_seen10_lrs(optimizer, 999, self.config["training"], _FAIR_POLICY, None)
        before = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(before["action_heads"], 1e-4)
        self.assertEqual(before["soft_prompt"], 1e-4)
        self.assertEqual(before["llm_lora"], 0.0)
        _update_seen10_lrs(optimizer, 1000, self.config["training"], _FAIR_POLICY, None)
        self.assertEqual({group["lr"] for group in optimizer.param_groups}, {1e-4})

    def test_parameter_roles_separate_frozen_vision_and_two_lora_backbones(self):
        self.assertEqual(_parameter_role("base_model.model.vlm.vision_tower.encoder.weight"), "vision_encoder")
        self.assertEqual(
            _parameter_role("base_model.model.vlm.language_model.model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight"),
            "llm_lora",
        )
        self.assertEqual(
            _parameter_role("base_model.model.transformer.blocks.0.attn.qkv.lora_A.default.weight"),
            "action_expert_lora",
        )
        self.assertEqual(_parameter_role("base_model.model.vlm.image_projection"), "vision_language_connector")

    def test_pruning_caps_physical_checkpoints_while_preserving_best_and_last(self):
        class Accelerator:
            is_main_process = True

            def wait_for_everyone(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            names = [f"step_{step:08d}" for step in (100, 200, 300, 400, 500, 600)]
            for name in names:
                (checkpoints / name).mkdir()
            (checkpoints / "best").symlink_to(names[0])
            (checkpoints / "last").symlink_to(names[-1])

            _prune_periodic_checkpoints(Accelerator(), root, 5)

            remaining = sorted(path.name for path in checkpoints.iterdir() if path.is_dir() and not path.is_symlink())
            self.assertEqual(len(remaining), 5)
            self.assertIn(names[0], remaining)
            self.assertIn(names[-1], remaining)

    def test_peft_merge_and_trainable_sidecar_preserve_all_trainable_values(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft is unavailable")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2, bias=True)
                self.head = torch.nn.Linear(3, 2, bias=False)
                self.image_projection = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, value):
                return self.linear(value) + self.head(value)

        model = get_peft_model(
            TinyModel(),
            LoraConfig(
                r=2,
                lora_alpha=4,
                bias="none",
                target_modules=["linear"],
                modules_to_save=["head"],
            ),
        )
        model.base_model.model.image_projection.requires_grad = True
        with torch.no_grad():
            model.base_model.model.linear.lora_A.default.weight.fill_(0.25)
            model.base_model.model.linear.lora_B.default.weight.fill_(0.5)
            model.base_model.model.head.modules_to_save.default.weight.fill_(2.0)
            model.base_model.model.image_projection.fill_(3.0)

        merged = _merged_inference_state(model)
        reference = deepcopy(model).merge_and_unload().state_dict()
        self.assertEqual(set(merged), set(reference))
        for name in reference:
            self.assertTrue(torch.equal(merged[name], reference[name]), name)

        expected_trainable = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            metadata = _save_full_resume_state(model, checkpoint)
            self.assertGreater(metadata["parameter_count"], 0)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.zero_()
            restored = _restore_full_resume_state(model, checkpoint)
            self.assertTrue(restored["restored"])
        actual = dict(model.named_parameters())
        for name, expected in expected_trainable.items():
            self.assertTrue(torch.equal(actual[name], expected), name)


if __name__ == "__main__":
    unittest.main()
