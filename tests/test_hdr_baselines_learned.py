"""
Trainable reference architectures — `hdr_baselines.learned`.

Why this file matters
─────────────────────
These nets exist so that "our model wins" is a claim against comparable
networks rather than only against classical filters. That makes two things
matter:

* **They must be trainable, and demonstrably so.** A shape-correct network
  whose gradients do not reach its parameters would sit in a comparison
  table looking like a fair opponent. Each is checked to produce finite
  gradients for every parameter and to reduce a loss over a few steps.
* **They must handle the shapes the data actually has.** Full frames are
  not multiples of the internal downsampling factors, so each architecture
  pads and crops internally; that is tested at awkward sizes.

Untrained scores are a smoke test, not a result — hence `trainable`, which
the runner uses to label them.
"""

import pytest
import torch

from hdr_baselines.learned import (
    DemosaicNetJDD,
    DnCNNJDD,
    LearnedBaseline,
    RestormerLiteJDD,
    UNetJDD,
)

ARCHITECTURES = (DnCNNJDD, UNetJDD, DemosaicNetJDD, RestormerLiteJDD)


def tiny(cls, **kwargs):
    """The smallest configuration of each that still exercises every path."""
    if cls is DnCNNJDD:
        return cls(width=8, depth=3, **kwargs)
    if cls is UNetJDD:
        return cls(base=8, levels=2, **kwargs)
    if cls is DemosaicNetJDD:
        return cls(width=8, depth=3, **kwargs)
    return cls(dim=8, num_blocks=1, heads=2, **kwargs)


@pytest.fixture
def inputs():
    return torch.rand(2, 4, 16, 16), torch.rand(2, 1, 16, 16)


class TestContract:
    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_the_shapes_match_the_project_contract(self, cls, inputs):
        x, snr = inputs
        model = tiny(cls)
        model.eval()
        with torch.no_grad():
            blended, experts, gates = model(x, snr)
        assert blended.shape == (2, 3, 32, 32)
        assert experts.shape == (2, 1, 3, 32, 32)
        assert gates.shape == (2, 1, 32, 32)

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_they_are_marked_trainable(self, cls):
        assert tiny(cls).trainable is True

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_the_parameter_count_is_reported(self, cls):
        """A comparison has to state it; leaving it implicit invites unfairness."""
        assert tiny(cls).num_parameters > 0

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_eval_mode_clamps_the_output(self, cls, inputs):
        x, snr = inputs
        model = tiny(cls)
        model.eval()
        with torch.no_grad():
            out = model(x, snr)[0]
        assert float(out.max()) <= 1.0 and float(out.min()) > 0.0

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_training_mode_leaves_headroom(self, cls, inputs):
        """The upper clamp would kill the gradient above 1.0."""
        x, snr = inputs
        model = tiny(cls)
        model.train()
        out = model(x, snr)[0]
        assert out.requires_grad

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_output_is_finite(self, cls, inputs):
        x, snr = inputs
        model = tiny(cls)
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(x, snr)[0]).all()

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_repr_states_the_size(self, cls):
        assert "params=" in tiny(cls).extra_repr()


class TestAwkwardShapes:
    @pytest.mark.parametrize("cls", ARCHITECTURES)
    @pytest.mark.parametrize("h,w", [(24, 40), (18, 22), (8, 8), (13, 17)])
    def test_internal_padding_preserves_the_output_size(self, cls, h, w):
        """Full frames are not multiples of the downsampling factors."""
        model = tiny(cls)
        model.eval()
        x = torch.rand(1, 4, h, w)
        snr = torch.rand(1, 1, h, w)
        with torch.no_grad():
            out = model(x, snr)[0]
        assert out.shape == (1, 3, h * 2, w * 2)

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_a_single_image_batch_works(self, cls):
        model = tiny(cls)
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 4, 16, 16), torch.rand(1, 1, 16, 16))[0]
        assert out.shape == (1, 3, 32, 32)


class TestTrainability:
    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_gradients_reach_every_parameter(self, cls, inputs):
        """A dead branch would look like a fair opponent and never learn."""
        x, snr = inputs
        model = tiny(cls)
        model.train()
        target = torch.rand(2, 3, 32, 32)
        loss = (model(x, snr)[0] - target).abs().mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{cls.name}: no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"{cls.name}: bad grad {name}"

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    @pytest.mark.slow
    def test_a_few_steps_reduce_the_loss(self, cls):
        torch.manual_seed(0)
        model = tiny(cls)
        model.train()
        x = torch.rand(2, 4, 16, 16)
        snr = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 3, 32, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        first = None
        for step in range(12):
            opt.zero_grad()
            loss = (model(x, snr)[0] - target).abs().mean()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
        assert float(loss) < first


class TestSNRAblation:
    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_the_snr_map_can_be_fed_in(self, cls, inputs):
        x, snr = inputs
        model = tiny(cls, use_snr=True)
        model.eval()
        with torch.no_grad():
            assert model(x, snr)[0].shape == (2, 3, 32, 32)

    @pytest.mark.parametrize("cls", ARCHITECTURES)
    def test_omitting_it_is_an_error_not_a_silent_zero(self, cls, inputs):
        x, _ = inputs
        model = tiny(cls, use_snr=True)
        with pytest.raises(ValueError, match="use_snr=True"):
            model(x, None)

    def test_it_changes_the_input_width(self):
        assert tiny(DnCNNJDD).in_channels == 4
        assert tiny(DnCNNJDD, use_snr=True).in_channels == 5

    def test_demosaicnet_builds_its_skip_from_the_cfa_only(self, inputs):
        """The SNR channel must not leak into the sparse-RGB skip."""
        x, snr = inputs
        model = tiny(DemosaicNetJDD, use_snr=True)
        model.eval()
        with torch.no_grad():
            a = model(x, snr)[0]
            b = model(x, torch.zeros_like(snr))[0]
        assert a.shape == b.shape


class TestValidation:
    def test_dncnn_rejects_a_degenerate_depth(self):
        with pytest.raises(ValueError, match="depth"):
            DnCNNJDD(depth=1)

    def test_dncnn_rejects_a_degenerate_width(self):
        with pytest.raises(ValueError, match="width"):
            DnCNNJDD(width=0)

    def test_unet_rejects_zero_levels(self):
        with pytest.raises(ValueError, match="levels"):
            UNetJDD(levels=0)

    def test_unet_rejects_a_tiny_base(self):
        with pytest.raises(ValueError, match="base"):
            UNetJDD(base=2)

    def test_demosaicnet_rejects_a_degenerate_depth(self):
        with pytest.raises(ValueError, match="depth"):
            DemosaicNetJDD(depth=1)

    def test_restormer_lite_rejects_zero_blocks(self):
        with pytest.raises(ValueError, match="num_blocks"):
            RestormerLiteJDD(num_blocks=0)

    def test_the_base_class_body_must_be_implemented(self):
        class Incomplete(LearnedBaseline):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete()(torch.rand(1, 4, 8, 8), None)
