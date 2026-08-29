"""
Tone curves — `hdr_eval.tonemap`.

Why this file matters
─────────────────────
Every PSNR number this project reports is a PSNR *through a curve*, and
the curve's constant is part of the number. mu=5000 is what the training
loss, the existing test script and the W&B curves all use; a run that
quietly used a different mu would produce numbers that look comparable
and are not. So the default is pinned here, along with the invertibility
that makes the curve a reparameterisation rather than a lossy step.
"""

import math

import pytest
import torch

from hdr_eval import tonemap as T


@pytest.fixture
def values():
    return torch.linspace(0.0, 1.0, 257)


class TestMuLaw:
    def test_the_default_mu_matches_training(self):
        """5000 is the constant train, test and W&B all assume."""
        assert T.MU_DEFAULT == 5000.0

    def test_it_maps_the_unit_interval_onto_itself(self, values):
        out = T.mu_law(values)
        assert float(out[0]) == 0.0
        assert abs(float(out[-1]) - 1.0) < 1e-6

    def test_it_is_monotonic(self, values):
        out = T.mu_law(values)
        assert torch.all(out[1:] >= out[:-1])

    def test_it_lifts_the_shadows_hard(self):
        """The whole point: 1% of full scale becomes a third of the range."""
        out = float(T.mu_law(torch.tensor(0.01)))
        assert 0.3 < out < 0.6

    def test_the_inverse_round_trips(self, values):
        assert torch.allclose(T.mu_law_inverse(T.mu_law(values)), values,
                              atol=1e-6)

    def test_it_matches_the_reference_formula(self):
        x = torch.tensor(0.37)
        expected = math.log1p(5000 * 0.37) / math.log1p(5000)
        assert abs(float(T.mu_law(x)) - expected) < 1e-6

    def test_it_agrees_with_the_training_script(self):
        from train_A100_MoE_two_phase import hdr_tonemap
        x = torch.rand(64)
        assert torch.allclose(T.mu_law(x, 5000), hdr_tonemap(x, 5000))

    def test_it_agrees_with_the_test_script(self):
        import test_dual_MoE_two_phase as ev
        x = torch.rand(64)
        assert torch.allclose(T.mu_law(x, 5000), ev.hdr_tonemap(x, 5000))

    @pytest.mark.parametrize("mu", [0.0, -1.0])
    def test_a_non_positive_mu_is_rejected(self, mu):
        with pytest.raises(ValueError, match="mu must be positive"):
            T.mu_law(torch.rand(4), mu)


class TestOtherCurves:
    def test_reinhard_never_saturates(self):
        assert float(T.reinhard(torch.tensor(1e6))) < 1.0

    def test_reinhard_is_monotonic(self, values):
        out = T.reinhard(values)
        assert torch.all(out[1:] >= out[:-1])

    def test_extended_reinhard_maps_the_white_point_to_one(self):
        out = float(T.reinhard_extended(torch.tensor(1.0), white_point=1.0))
        assert abs(out - 1.0) < 1e-6

    def test_extended_reinhard_rejects_a_bad_white_point(self):
        with pytest.raises(ValueError, match="white_point"):
            T.reinhard_extended(torch.rand(4), white_point=0.0)

    def test_gamma_round_trips(self, values):
        assert torch.allclose(T.gamma_decode(T.gamma_encode(values)), values,
                              atol=1e-6)

    @pytest.mark.parametrize("fn", [T.gamma_encode, T.gamma_decode])
    def test_gamma_rejects_a_non_positive_exponent(self, fn):
        with pytest.raises(ValueError, match="gamma"):
            fn(torch.rand(4), gamma=0.0)

    def test_log_encoding_spans_the_unit_interval(self, values):
        out = T.log_encode(values)
        assert abs(float(out[0])) < 1e-6
        assert abs(float(out[-1]) - 1.0) < 1e-6

    def test_log_encoding_rejects_a_non_positive_eps(self):
        with pytest.raises(ValueError, match="eps"):
            T.log_encode(torch.rand(4), eps=0.0)

    def test_identity_is_a_no_op(self, values):
        assert torch.equal(T.identity(values), values)

    def test_curves_preserve_shape_and_dtype(self):
        x = torch.rand(2, 3, 8, 8, dtype=torch.float64)
        for name in T.list_tonemaps():
            out = T.tonemap(x, name)
            assert out.shape == x.shape and out.dtype == x.dtype


class TestRegistry:
    def test_every_registered_name_resolves(self):
        for name in T.list_tonemaps():
            assert callable(T.get_tonemap(name))

    def test_aliases_point_at_the_same_function(self):
        assert T.get_tonemap("mu") is T.get_tonemap("mu_law")
        assert T.get_tonemap("linear") is T.get_tonemap("none")

    def test_unknown_name_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="reinhard"):
            T.get_tonemap("filmic")

    def test_tonemap_forwards_keyword_arguments(self):
        x = torch.tensor(0.5)
        assert not torch.equal(T.tonemap(x, "mu", mu=10.0),
                               T.tonemap(x, "mu", mu=5000.0))
