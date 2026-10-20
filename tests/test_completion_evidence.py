import unittest
import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import ppo_policy_loss, compute_gae
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss
from task5_feedback.score_perturbations import _summarize_pairwise


class EvidenceTests(unittest.TestCase):
    def test_dpo_reference_adjustment(self):
        p = torch.tensor([2., -1.])
        z = torch.zeros(2)
        loss, diag = dpo_loss(p, z, z, z, .1)
        self.assertTrue(torch.allclose(loss, -torch.nn.functional.logsigmoid(.1*p).mean()))
        self.assertEqual(float(diag['preference_accuracy']), .5)

    def test_signed_ppo_clipping(self):
        ratio = torch.tensor([[1.5, .5]])
        loss, _, clipped = ppo_policy_loss(ratio.log(), torch.zeros_like(ratio),
            torch.tensor([[1., -1.]]), torch.ones_like(ratio), .2)
        self.assertAlmostEqual(float(loss), -.2, places=6)
        self.assertEqual(float(clipped), 1.)

    def test_masked_gae(self):
        reward = torch.tensor([[1., 2., 0.]])
        a, _ = compute_gae(reward, torch.zeros_like(reward),
            torch.tensor([[1., 1., 0.]]), gamma=1., lam=1.)
        self.assertTrue(torch.allclose(a, torch.tensor([[3., 2., 0.]])))

    def test_constant_normalization_gradient(self):
        mask = torch.tensor([[1., 0., 0., 0.], [1., 1., 1., 1.]])
        for norm, expected in [('grpo', [0.5, 0.125]), ('dr_grpo', [0.125, 0.125])]:
            new = torch.zeros_like(mask, requires_grad=True)
            z = torch.zeros_like(mask)
            loss, _ = grpo_policy_loss(new,z,torch.ones(2),mask,z,.2,0.,norm,4)
            grad = torch.autograd.grad(loss,new)[0].abs()
            self.assertAlmostEqual(float(grad[0,0]),expected[0])
            self.assertAlmostEqual(float(grad[1,0]),expected[1])
            self.assertEqual(float(grad[0,1]),0.)

    def test_grpo_uses_within_group_statistics(self):
        a = group_relative_advantages(torch.tensor([-1.,1.,10.,12.]),torch.tensor([0,0,1,1]))
        self.assertTrue(torch.allclose(a,torch.tensor([-1.,1.,-1.,1.]),atol=2e-6))

    def test_reasoning_tie_is_not_sensitivity(self):
        rows = [{'comparison':'reasoning_sensitivity', 'preferred_variant':'tie',
            'left_variant':'clean_correct','right_variant':'corrupt_reasoning_correct_final',
            'left_exact_reward':1.,'right_exact_reward':1.}]
        r = _summarize_pairwise(rows)['reasoning_sensitivity']
        self.assertEqual(r['judge_exact_agreement_rate'],1.)
        self.assertEqual(r['ai_better_rate'],0.)
        self.assertEqual(r['exact_better_rate'],0.)


if __name__ == '__main__':
    unittest.main()
