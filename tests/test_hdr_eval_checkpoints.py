"""
Rebuilding a trained model — `hdr_eval.checkpoints`.

Why this file matters
─────────────────────
`train_A100_MoE_two_phase.py` stores `mode`, `model_kwargs` and
`num_experts` in every checkpoint precisely so the benchmark does not have
to be kept in sync by hand. The existing loader in
test_dual_MoE_two_phase.py does not use the stored expert count — the line
reads

    num_experts = 2 #ckpt.get("num_experts", fallback_num_experts)

so a 3-expert checkpoint either fails to load or, where the shapes happen
to line up, loads while being described wrongly in the results. This
module reads the stored value, and the test below is the one that would
have caught the regression: a 3-expert checkpoint must come back with
three experts.
"""

import pytest
import torch

from HDR_model_hybrid_Teacher import build_denoiser
from helpers import make_checkpoint
from hdr_eval.checkpoints import (
    CheckpointInfo,
    load_checkpoint,
    load_model_from_checkpoint,
)


@pytest.fixture
def tiny_kwargs_local():
    return {"dim": 8, "num_blocks": [1, 1, 1, 1], "num_refinement_blocks": 1,
            "heads": [1, 1, 1, 1], "se_reduction": 8}


def write_checkpoint(path, mode, num_experts, kwargs):
    model = build_denoiser(mode, num_experts=num_experts, **kwargs)
    return make_checkpoint(path, model, mode=mode, num_experts=num_experts,
                           model_kwargs=kwargs)


class TestReadingMetadata:
    def test_it_reports_what_the_checkpoint_says(self, tmp_path, tiny_kwargs_local):
        path = write_checkpoint(tmp_path / "c.pth", "moe", 3, tiny_kwargs_local)
        _, info = load_checkpoint(path)
        assert info.mode == "moe" and info.num_experts == 3
        assert info.has_metadata and info.model_kwargs["dim"] == 8

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            load_checkpoint(str(tmp_path / "absent.pth"))

    def test_a_non_checkpoint_file_is_reported(self, tmp_path):
        path = str(tmp_path / "raw.pth")
        torch.save(torch.rand(4), path)
        with pytest.raises(ValueError, match="checkpoint dict"):
            load_checkpoint(path)

    def test_info_is_json_friendly_and_readable(self, tmp_path, tiny_kwargs_local):
        path = write_checkpoint(tmp_path / "c.pth", "moe", 2, tiny_kwargs_local)
        _, info = load_checkpoint(path)
        import json
        json.dumps(info.to_dict())
        assert "moe" in repr(info)


class TestRebuildingTheModel:
    @pytest.mark.parametrize("mode,num_experts",
                             [("moe", 2), ("moe", 3), ("moe", 4),
                              ("dual", 2), ("single", 1)])
    def test_it_round_trips_every_architecture(self, tmp_path, tiny_kwargs_local,
                                               mode, num_experts):
        path = write_checkpoint(tmp_path / "c.pth", mode, num_experts,
                                tiny_kwargs_local)
        model, info = load_model_from_checkpoint(path)
        assert info.mode == mode
        assert model.num_experts == num_experts

    def test_the_stored_expert_count_is_used(self, tmp_path, tiny_kwargs_local):
        """
        The specific regression: hardcoding 2 here would either raise on
        the state_dict load or mislabel the model in every report.
        """
        path = write_checkpoint(tmp_path / "c.pth", "moe", 3, tiny_kwargs_local)
        model, info = load_model_from_checkpoint(path)
        assert model.num_experts == 3 and info.num_experts == 3

    def test_the_weights_actually_load(self, tmp_path, tiny_kwargs_local):
        original = build_denoiser("moe", num_experts=2, **tiny_kwargs_local)
        with torch.no_grad():
            for p in original.parameters():
                p.add_(0.01)
        path = make_checkpoint(tmp_path / "c.pth", original, mode="moe",
                               num_experts=2, model_kwargs=tiny_kwargs_local)
        loaded, _ = load_model_from_checkpoint(path)
        for a, b in zip(original.parameters(), loaded.parameters()):
            assert torch.equal(a, b)

    def test_the_model_comes_back_in_eval_mode(self, tmp_path, tiny_kwargs_local):
        """Eval mode is what enables the upper output clamp."""
        path = write_checkpoint(tmp_path / "c.pth", "moe", 2, tiny_kwargs_local)
        model, _ = load_model_from_checkpoint(path)
        assert not model.training

    def test_it_is_marked_as_carrying_real_weights(self, tmp_path, tiny_kwargs_local):
        path = write_checkpoint(tmp_path / "c.pth", "moe", 2, tiny_kwargs_local)
        model, _ = load_model_from_checkpoint(path)
        assert model.loaded_checkpoint is True

    def test_a_compiled_checkpoint_loads(self, tmp_path, tiny_kwargs_local):
        """torch.compile prefixes every key with _orig_mod."""
        model = build_denoiser("moe", num_experts=2, **tiny_kwargs_local)
        state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        path = str(tmp_path / "c.pth")
        torch.save({"mode": "moe", "num_experts": 2,
                    "model_kwargs": tiny_kwargs_local,
                    "model_state_dict": state}, path)
        loaded, _ = load_model_from_checkpoint(path)
        assert loaded.num_experts == 2

    def test_a_legacy_checkpoint_uses_the_fallbacks(self, tmp_path,
                                                    tiny_kwargs_local):
        model = build_denoiser("single", num_experts=1, **tiny_kwargs_local)
        path = str(tmp_path / "legacy.pth")
        torch.save({"model_state_dict": model.state_dict()}, path)
        loaded, info = load_model_from_checkpoint(
            path, fallback_kwargs=tiny_kwargs_local, fallback_num_experts=1,
            fallback_mode="single")
        assert not info.has_metadata
        assert loaded.num_experts == 1

    def test_a_legacy_checkpoint_with_the_wrong_fallback_mode_fails_loudly(
            self, tmp_path, tiny_kwargs_local):
        """
        There is nothing in a pre-metadata checkpoint to infer the mode
        from, so a wrong guess must surface as a state_dict error rather
        than as a model quietly built from the wrong architecture.
        """
        model = build_denoiser("single", num_experts=1, **tiny_kwargs_local)
        path = str(tmp_path / "legacy.pth")
        torch.save({"model_state_dict": model.state_dict()}, path)
        with pytest.raises(RuntimeError, match="state_dict"):
            load_model_from_checkpoint(path, fallback_kwargs=tiny_kwargs_local,
                                       fallback_mode="dual")

    def test_the_loaded_model_produces_the_expected_shapes(self, tmp_path,
                                                           tiny_kwargs_local):
        path = write_checkpoint(tmp_path / "c.pth", "moe", 2, tiny_kwargs_local)
        model, _ = load_model_from_checkpoint(path)
        x = torch.rand(1, 4, 16, 16)
        snr = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            blended, experts, gates = model(x, snr)
        assert blended.shape == (1, 3, 32, 32)
        assert experts.shape == (1, 2, 3, 32, 32)
        assert gates.shape == (1, 2, 32, 32)
