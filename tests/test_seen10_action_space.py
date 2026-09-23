import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.action_hub import (
    ACTION_REGISTRY,
    AutoActionSpace,
    OfficialAutoActionSpace,
    build_action_space,
    pad_action_to_model_dim,
)

try:
    from csgo_seen10.model import configure_seen10, pad_seen10_action, validate_seen10_model
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent optional deps
    _CSGO_MODEL_IMPORT_ERROR = error
    pad_seen10_action = pad_action_to_model_dim
else:
    _CSGO_MODEL_IMPORT_ERROR = None


class AutoActionSpaceSeen10Test(unittest.TestCase):
    def test_model_width_noise_matches_zero_padded_real_width_action(self):
        space = AutoActionSpace(real_dim=5, max_dim=20)
        proprio = torch.zeros(3, 20)
        noisy_model_width = torch.randn(3, 1, 20)

        _, preprocessed_noise = space.preprocess(proprio, noisy_model_width)
        _, preprocessed_real_width = space.preprocess(proprio, noisy_model_width[..., :5])

        self.assertEqual(tuple(preprocessed_noise.shape), (3, 1, 20))
        self.assertEqual(int(torch.count_nonzero(preprocessed_noise[..., 5:])), 0)
        self.assertTrue(torch.equal(preprocessed_noise, preprocessed_real_width))


class OfficialAutoActionSpaceSeen10Test(unittest.TestCase):
    def test_explicit_pad_helper_keeps_external_target_and_model_width(self):
        target = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5)
        padded = pad_seen10_action(target)

        self.assertEqual(tuple(target.shape), (2, 1, 5))
        self.assertEqual(tuple(padded.shape), (2, 1, 20))
        self.assertTrue(torch.equal(padded[..., :5], target))
        self.assertEqual(int(torch.count_nonzero(padded[..., 5:])), 0)

        # A model-width action is already prepared and must pass by identity.
        self.assertIs(pad_seen10_action(padded), padded)

    def test_official_preprocess_preserves_model_width_dummy_channels(self):
        space = OfficialAutoActionSpace(real_dim=5, max_dim=20)
        proprio = torch.zeros(2, 20)
        action = torch.randn(2, 1, 20)
        action[..., 5:] = torch.arange(15, dtype=action.dtype)

        returned_proprio, prepared = space.preprocess(proprio, action)
        self.assertIs(returned_proprio, proprio)
        self.assertIs(prepared, action)
        self.assertTrue(torch.equal(prepared, action))

    def test_official_loss_uses_only_first_five_dimensions(self):
        space = build_action_space("official_auto", real_dim=5, max_dim=20)
        pred = torch.zeros(2, 1, 20, requires_grad=True)
        target = torch.zeros(2, 1, 5)
        values = torch.cat((torch.ones(2, 1, 5), torch.full((2, 1, 15), 123.0)), dim=-1)
        pred = pred + values

        loss = space.compute_loss(pred, target)["joints_loss"]
        self.assertTrue(torch.equal(loss, torch.tensor(100.0)))

        padded_target = pad_seen10_action(target)
        loss_with_padded_target = space.compute_loss(pred, padded_target)["joints_loss"]
        self.assertTrue(torch.equal(loss, loss_with_padded_target))
        gradient = torch.autograd.grad(loss, pred)[0]
        self.assertGreater(int(torch.count_nonzero(gradient[..., :5])), 0)
        self.assertEqual(int(torch.count_nonzero(gradient[..., 5:])), 0)

    def test_official_postprocess_exposes_only_real_dimensions(self):
        space = OfficialAutoActionSpace(real_dim=5, max_dim=20)
        action = torch.randn(2, 1, 20)
        output = space.postprocess(action)
        self.assertEqual(tuple(output.shape), (2, 1, 5))
        self.assertTrue(torch.equal(output, action[..., :5]))

    def test_legacy_reset_dummy_alias_preserves_old_behavior(self):
        self.assertIs(ACTION_REGISTRY["legacy_reset_dummy"], AutoActionSpace)
        space = build_action_space("legacy_reset_dummy", real_dim=5, max_dim=20)
        proprio = torch.zeros(2, 20)
        action = torch.randn(2, 1, 20)

        _, prepared = space.preprocess(proprio, action)
        self.assertEqual(int(torch.count_nonzero(prepared[..., 5:])), 0)


class Seen10ModelContractTest(unittest.TestCase):
    @unittest.skipIf(_CSGO_MODEL_IMPORT_ERROR is not None, "CSGO model dependencies are unavailable")
    def test_action_mode_is_persisted_in_config(self):
        config = configure_seen10(XVLAConfig())
        self.assertEqual(config.action_mode, "official_auto")
        self.assertEqual(config.to_dict()["action_mode"], "official_auto")
        legacy = configure_seen10(XVLAConfig(), action_mode="legacy_reset_dummy")
        self.assertEqual(legacy.action_mode, "legacy_reset_dummy")

    @unittest.skipIf(_CSGO_MODEL_IMPORT_ERROR is not None, "CSGO model dependencies are unavailable")
    def test_official_forward_rejects_unpadded_target_before_noise(self):
        fake_model = SimpleNamespace(
            action_mode="official_auto",
            action_space=SimpleNamespace(dim_action=20),
        )
        with self.assertRaisesRegex(ValueError, "padded before XVLA.forward"):
            XVLA.forward(
                fake_model,
                input_ids=torch.zeros(2, 3, dtype=torch.long),
                image_input=torch.zeros(2, 2, 3, 4, 4),
                image_mask=torch.ones(2, 2, dtype=torch.bool),
                domain_id=torch.zeros(2, dtype=torch.long),
                proprio=torch.zeros(2, 20),
                action=torch.zeros(2, 1, 5),
            )

    @unittest.skipIf(_CSGO_MODEL_IMPORT_ERROR is not None, "CSGO model dependencies are unavailable")
    def test_official_forward_samples_noise_after_target_is_twenty_dimensional(self):
        captured = {}

        class Transformer:
            def __call__(self, *, action_with_noise, **kwargs):
                captured["model_input"] = action_with_noise
                return action_with_noise

        fake_model = SimpleNamespace(
            action_mode="official_auto",
            action_space=OfficialAutoActionSpace(real_dim=5, max_dim=20),
            forward_vlm=lambda *args, **kwargs: {},
            transformer=Transformer(),
        )
        target = pad_seen10_action(torch.zeros(2, 1, 5))

        def fake_randn_like(value):
            captured["noise_shape"] = tuple(value.shape)
            return torch.ones_like(value)

        with (
            patch("models.modeling_xvla.torch.randn_like", side_effect=fake_randn_like),
            patch("models.modeling_xvla.torch.rand", return_value=torch.tensor([0.25])),
        ):
            loss = XVLA.forward(
                fake_model,
                input_ids=torch.zeros(2, 3, dtype=torch.long),
                image_input=torch.zeros(2, 2, 3, 4, 4),
                image_mask=torch.ones(2, 2, dtype=torch.bool),
                domain_id=torch.zeros(2, dtype=torch.long),
                proprio=torch.zeros(2, 20),
                action=target,
            )["joints_loss"]

        self.assertEqual(captured["noise_shape"], (2, 1, 20))
        self.assertEqual(tuple(captured["model_input"].shape), (2, 1, 20))
        self.assertGreater(int(torch.count_nonzero(captured["model_input"][..., 5:])), 0)
        self.assertGreater(float(loss), 0.0)

    @unittest.skipIf(_CSGO_MODEL_IMPORT_ERROR is not None, "CSGO model dependencies are unavailable")
    def test_official_generation_preserves_model_only_channels_between_steps(self):
        model_inputs = []

        class Transformer:
            def __call__(self, *, action_with_noise, **kwargs):
                model_inputs.append(action_with_noise.clone())
                result = torch.zeros_like(action_with_noise)
                result[..., 5:] = float(len(model_inputs))
                return result

        fake_model = SimpleNamespace(
            eval=lambda: None,
            forward_vlm=lambda *args, **kwargs: {},
            action_space=OfficialAutoActionSpace(real_dim=5, max_dim=20),
            transformer=Transformer(),
            num_actions=1,
        )
        with patch("models.modeling_xvla.torch.randn", return_value=torch.zeros(2, 1, 20)):
            prediction = XVLA.generate_actions(
                fake_model,
                input_ids=torch.zeros(2, 3, dtype=torch.long),
                image_input=torch.zeros(2, 2, 3, 4, 4),
                image_mask=torch.ones(2, 2, dtype=torch.bool),
                domain_id=torch.zeros(2, dtype=torch.long),
                proprio=torch.zeros(2, 20),
                steps=2,
            )

        self.assertEqual(tuple(prediction.shape), (2, 1, 5))
        self.assertEqual(len(model_inputs), 2)
        self.assertGreater(int(torch.count_nonzero(model_inputs[1][..., 5:])), 0)

    @unittest.skipIf(_CSGO_MODEL_IMPORT_ERROR is not None, "CSGO model dependencies are unavailable")
    def test_loader_contract_checks_encoder_decoder_and_proprio(self):
        model = SimpleNamespace(
            config=SimpleNamespace(action_mode="official_auto"),
            num_actions=1,
            action_space=SimpleNamespace(real_dim=5, dim_action=20),
            transformer=SimpleNamespace(
                action_encoder=SimpleNamespace(input_size=72),
                action_decoder=SimpleNamespace(output_size=20),
            ),
            use_proprio=False,
        )
        self.assertIs(validate_seen10_model(model), model)


if __name__ == "__main__":
    unittest.main()
