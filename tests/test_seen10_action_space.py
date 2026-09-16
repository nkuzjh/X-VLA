import unittest

import torch

from models.action_hub import AutoActionSpace


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


if __name__ == "__main__":
    unittest.main()
