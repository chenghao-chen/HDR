"""
The CUDA/XPU/CPU abstraction — hdr_platform/accelerator.py.

No CI host has both an NVIDIA and an Intel GPU, so the interesting cases
cannot be produced for real. They are reached here by substituting a fake
backend module for ``torch.cuda`` / ``torch.xpu``, which is enough because
the class only ever touches a handful of names on it. What that buys is
coverage of the Aurora paths from a Polaris login node — the alternative is
discovering them inside a queued job.

The failure this file is guarding against, concretely: a literal ``"cuda"``
reaching torch on an Intel GPU. It does not degrade, it raises, and it raises
after the model and the dataset have loaded.
"""

import contextlib

import pytest
import torch

from hdr_platform.accelerator import (
    Accelerator, available_backends, get_accelerator, resolve_device,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fakes
# ═════════════════════════════════════════════════════════════════════════════
class FakeProperties:
    def __init__(self, name, total_memory):
        self.name = name
        self.total_memory = total_memory


class FakeBackend:
    """
    Stand-in for torch.cuda / torch.xpu.

    Records the calls that have to reach the driver (synchronize, cache
    eviction, seeding) so tests can assert they were actually made rather
    than swallowed by a bare ``except``.
    """

    def __init__(self, name="Fake GPU", total_memory=64 * 10 ** 9, count=1,
                 raises=()):
        self._props = FakeProperties(name, total_memory)
        self._count = count
        self._raises = set(raises)
        self.calls = []

    def _record(self, what):
        self.calls.append(what)
        if what in self._raises:
            raise RuntimeError(f"{what} exploded")

    def get_device_properties(self, index): self._record("props"); return self._props
    def device_count(self): self._record("count"); return self._count
    def synchronize(self): self._record("synchronize")
    def empty_cache(self): self._record("empty_cache")
    def reset_peak_memory_stats(self): self._record("reset_peak")
    def memory_allocated(self, i=0): self._record("mem"); return 1_500_000_000
    def max_memory_allocated(self, i=0): self._record("peak"); return 3_000_000_000
    def manual_seed_all(self, seed): self._record(f"seed_all:{seed}")


def fake_accelerator(kind, backend=None, **kw):
    """An Accelerator whose backend module is a FakeBackend."""
    acc = Accelerator.__new__(Accelerator)
    acc.device = torch.device(f"{kind}:0")
    acc.kind = kind
    acc._backend = backend if backend is not None else FakeBackend(**kw)
    return acc


# ═════════════════════════════════════════════════════════════════════════════
# Device resolution
# ═════════════════════════════════════════════════════════════════════════════
class TestResolveDevice:
    def test_cpu_is_always_resolvable(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_picks_a_real_backend_and_never_raises(self):
        """
        `auto` runs on the login node, in a job, and on a laptop. Whatever it
        returns must be a device that exists here.
        """
        dev = resolve_device("auto")
        assert dev.type in ("cuda", "xpu", "cpu")
        torch.zeros(1, device=dev)

    def test_none_and_auto_agree(self):
        assert resolve_device(None) == resolve_device("auto")

    def test_hdr_device_is_honoured(self, monkeypatch):
        """How a submit script pins a job to one specific tile."""
        monkeypatch.setenv("HDR_DEVICE", "cpu")
        assert resolve_device() == torch.device("cpu")

    def test_asking_for_an_absent_backend_fails_immediately(self):
        """
        The error has to arrive here, at resolution, not 40 lines later
        inside a .to() call with a stack trace that names neither the device
        nor the reason.
        """
        absent = "xpu" if not _xpu_here() else "cuda"
        if _backend_here(absent):
            pytest.skip(f"this host actually has {absent}")
        with pytest.raises(RuntimeError, match="no .* device is visible"):
            resolve_device(f"{absent}:0")

    def test_the_absent_backend_error_says_what_to_do(self):
        """
        Two very different fixes depending on the backend, and the message
        has to name the right one: submit a job (Polaris) versus load the
        frameworks module (Aurora).
        """
        if not _backend_here("xpu"):
            with pytest.raises(RuntimeError, match="frameworks"):
                resolve_device("xpu:0")
        if not _backend_here("cuda"):
            with pytest.raises(RuntimeError, match="login node|submit a job"):
                resolve_device("cuda:0")

    def test_available_backends_always_ends_in_cpu(self):
        """CPU is the guaranteed fallback, so it must always be offered."""
        backends = available_backends()
        assert backends[-1] == "cpu"
        assert len(set(backends)) == len(backends)


def _backend_here(kind):
    return kind in available_backends()


def _xpu_here():
    return _backend_here("xpu")


# ═════════════════════════════════════════════════════════════════════════════
# Autocast — the part that breaks on Aurora
# ═════════════════════════════════════════════════════════════════════════════
class TestAutocast:
    @pytest.mark.parametrize("kind", ["cuda", "xpu"])
    def test_it_names_the_devices_own_backend(self, kind, monkeypatch):
        """
        The bug this package was written to remove: `device_type="cuda"`
        hardcoded at every autocast site. On an Intel GPU that raises.
        """
        seen = []

        def record(*args, **kwargs):
            seen.append(kwargs.get("device_type", args[0] if args else None))
            return contextlib.nullcontext()

        monkeypatch.setattr(torch.amp, "autocast", record)
        with fake_accelerator(kind).autocast(torch.bfloat16):
            pass
        assert seen == [kind]

    def test_cpu_gets_no_autocast_at_all(self):
        """
        bf16 autocast on the CPU is legal and slower than fp32 for this
        model, so it is skipped rather than merely unsupported. A CPU smoke
        run should measure the same arithmetic a GPU run does.
        """
        with get_accelerator("cpu").autocast(torch.bfloat16):
            assert (torch.rand(4, 4) @ torch.rand(4, 4)).dtype == torch.float32

    def test_a_none_dtype_disables_it_on_any_backend(self):
        """`--no-amp` in the eval CLI has to work everywhere."""
        for kind in ("cuda", "xpu"):
            ctx = fake_accelerator(kind).autocast(None)
            assert isinstance(ctx, contextlib.nullcontext)

    @pytest.mark.gpu
    def test_it_really_engages_on_this_hosts_gpu(self, accelerator):
        with accelerator.autocast(torch.bfloat16):
            out = (torch.rand(8, 8, device=accelerator.device)
                   @ torch.rand(8, 8, device=accelerator.device))
        assert out.dtype == torch.bfloat16


# ═════════════════════════════════════════════════════════════════════════════
# Optimizer and DataLoader arguments
# ═════════════════════════════════════════════════════════════════════════════
class TestOptimizerKwargs:
    def test_fused_adam_is_offered_only_on_cuda(self):
        """
        `fused=True` is a CUDA-only fast path. Passing it on XPU raises at
        optimiser construction — several minutes into a run, after the
        dataset has loaded.
        """
        assert fake_accelerator("cuda").supports_fused_adam is True
        assert fake_accelerator("xpu").supports_fused_adam is False
        assert get_accelerator("cpu").supports_fused_adam is False

    def test_an_explicit_fused_request_is_still_dropped_off_cuda(self):
        """
        A caller passing fused=True by hand must not be able to break the
        Aurora run; the backend, not the caller, is authoritative here.
        """
        kwargs = fake_accelerator("xpu").optimizer_kwargs(lr=1e-4, fused=True)
        assert kwargs["fused"] is False
        assert kwargs["lr"] == 1e-4

    def test_other_arguments_pass_through_untouched(self):
        kwargs = fake_accelerator("cuda").optimizer_kwargs(
            lr=3e-4, betas=(0.9, 0.99))
        assert kwargs == {"fused": True, "lr": 3e-4, "betas": (0.9, 0.99)}

    def test_real_adam_accepts_what_this_produces(self):
        """The kwargs have to be legal, not merely plausible."""
        acc = get_accelerator("cpu")
        p = [torch.nn.Parameter(torch.zeros(2))]
        torch.optim.Adam(p, **acc.optimizer_kwargs(lr=1e-4, betas=(0.9, 0.999)))


class TestDataLoaderKwargs:
    def test_xpu_names_its_pin_memory_device(self):
        """
        DataLoader pins for CUDA by default. On Aurora a bare
        pin_memory=True gives up the transfer speedup without saying so, so
        the target device has to be named explicitly.
        """
        kwargs = fake_accelerator("xpu").dataloader_kwargs(num_workers=4)
        assert kwargs["pin_memory"] is True
        assert kwargs["pin_memory_device"] == "xpu"

    def test_cuda_does_not_need_the_extra_argument(self):
        kwargs = fake_accelerator("cuda").dataloader_kwargs(num_workers=4)
        assert kwargs["pin_memory"] is True
        assert "pin_memory_device" not in kwargs

    def test_cpu_does_not_pin(self):
        """Nothing to copy to; pinning would cost host memory for nothing."""
        kwargs = get_accelerator("cpu").dataloader_kwargs(num_workers=2)
        assert kwargs["pin_memory"] is False
        assert "pin_memory_device" not in kwargs

    def test_persistent_workers_only_with_workers(self):
        """
        persistent_workers=True with num_workers=0 is a ValueError from
        DataLoader, and num_workers=0 is what a debug run uses.
        """
        assert "persistent_workers" not in \
            get_accelerator("cpu").dataloader_kwargs(num_workers=0)
        assert fake_accelerator("cuda").dataloader_kwargs(
            num_workers=2)["persistent_workers"] is True

    def test_an_explicit_pin_memory_false_is_respected(self):
        kwargs = fake_accelerator("xpu").dataloader_kwargs(
            num_workers=4, pin_memory=False)
        assert kwargs["pin_memory"] is False
        assert "pin_memory_device" not in kwargs

    def test_real_dataloader_accepts_what_this_produces(self):
        from torch.utils.data import DataLoader, TensorDataset
        acc = get_accelerator("cpu")
        ds = TensorDataset(torch.zeros(4, 2))
        loader = DataLoader(ds, batch_size=2, **acc.dataloader_kwargs(num_workers=0))
        assert len(list(loader)) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Backend operations
# ═════════════════════════════════════════════════════════════════════════════
class TestBackendOperations:
    def test_synchronize_reaches_the_backend(self):
        """
        Every timing number in the benchmark depends on this actually
        happening: GPU work is queued asynchronously, so a swallowed
        synchronize turns wall-clock measurements into submission latency.
        """
        backend = FakeBackend()
        fake_accelerator("cuda", backend).synchronize()
        assert "synchronize" in backend.calls

    def test_synchronize_is_a_no_op_on_cpu(self):
        get_accelerator("cpu").synchronize()      # must not raise

    def test_memory_is_reported_in_gigabytes(self):
        acc = fake_accelerator("cuda")
        assert acc.memory_gb() == pytest.approx(1.5)
        assert acc.peak_memory_gb() == pytest.approx(3.0)
        assert acc.total_memory_gb() == pytest.approx(64.0)

    def test_seeding_prefers_the_all_devices_variant(self):
        backend = FakeBackend()
        fake_accelerator("cuda", backend).seed_all(21)
        assert "seed_all:21" in backend.calls

    def test_seeding_still_seeds_the_host_rng_without_a_backend(self):
        """
        On the CPU there is no device RNG, but torch.manual_seed must still
        run or a CPU smoke test stops being reproducible.
        """
        get_accelerator("cpu").seed_all(1234)
        a = torch.rand(3)
        get_accelerator("cpu").seed_all(1234)
        assert torch.equal(a, torch.rand(3))

    @pytest.mark.parametrize("failing", ["synchronize", "empty_cache",
                                         "reset_peak", "peak", "count", "props"])
    def test_a_backend_that_raises_does_not_take_the_job_down(self, failing):
        """
        These are all reporting or housekeeping calls. A driver that refuses
        one of them mid-run must not end an eight-hour job — the training
        loop's correctness does not depend on any of them.
        """
        acc = fake_accelerator("cuda", FakeBackend(raises=[failing]))
        acc.synchronize()
        acc.empty_cache()
        acc.reset_peak_memory()
        assert acc.peak_memory_gb() >= 0.0
        assert acc.device_count() >= 0
        assert isinstance(acc.name, str)

    def test_enable_fast_matmul_is_safe_on_every_backend(self):
        for kind in ("cuda", "xpu"):
            fake_accelerator(kind).enable_fast_matmul()
        get_accelerator("cpu").enable_fast_matmul()
        assert torch.get_float32_matmul_precision() == "high"

    def test_optimize_model_is_a_pass_through_without_ipex(self):
        """
        ipex.optimize is a performance nicety, not a correctness requirement.
        Its absence must return the model unchanged, and its presence must
        not change the call signature.
        """
        model = torch.nn.Linear(2, 2)
        assert fake_accelerator("cuda").optimize_model(model) is model
        assert get_accelerator("cpu").optimize_model(model) is model

        opt = torch.optim.Adam(model.parameters())
        m2, o2 = fake_accelerator("cuda").optimize_model(model, opt)
        assert (m2, o2) == (model, opt)


# ═════════════════════════════════════════════════════════════════════════════
# Identity and caching
# ═════════════════════════════════════════════════════════════════════════════
class TestIdentity:
    def test_is_gpu_distinguishes_the_three_kinds(self):
        assert fake_accelerator("cuda").is_gpu
        assert fake_accelerator("xpu").is_gpu
        assert not get_accelerator("cpu").is_gpu

    def test_index_defaults_to_zero_when_unspecified(self):
        acc = Accelerator(torch.device("cpu"))
        assert acc.index == 0

    def test_summary_names_the_device_and_its_memory(self):
        """This line is the header of every job log; it has to be readable."""
        text = fake_accelerator("xpu", name="Intel Max 1550").summary()
        assert "xpu:0" in text and "Intel Max 1550" in text and "64 GB" in text

    def test_cpu_summary_reports_the_thread_count(self):
        assert "cpu" in get_accelerator("cpu").summary()

    def test_accelerators_are_cached_per_device(self):
        """
        Probing IPEX loads the oneAPI runtime, which is slow. Constructing
        the accelerator twice must not pay for it twice.
        """
        assert get_accelerator("cpu") is get_accelerator("cpu")

    def test_repr_is_useful_in_a_traceback(self):
        assert "cpu" in repr(get_accelerator("cpu"))


# ═════════════════════════════════════════════════════════════════════════════
# On whatever GPU this host actually has
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.gpu
class TestOnRealHardware:
    def test_the_detected_backend_matches_the_detected_site(self, accelerator):
        """
        A mismatch means either detection is wrong or the wrong module is
        loaded — on Aurora, usually that `frameworks` failed and a CPU-only
        torch is in play.
        """
        from hdr_platform import detect_site, get_site
        site = get_site(detect_site())
        assert accelerator.kind == site.accelerator

    def test_it_can_actually_compute(self, accelerator):
        a = torch.randn(64, 64, device=accelerator.device)
        with accelerator.autocast(torch.bfloat16):
            c = a @ a
        accelerator.synchronize()
        assert torch.isfinite(c.float()).all()

    def test_memory_accounting_moves_when_memory_is_used(self, accelerator):
        accelerator.reset_peak_memory()
        big = torch.zeros(1024, 1024, 32, device=accelerator.device)
        accelerator.synchronize()
        assert accelerator.peak_memory_gb() > 0.0
        del big

    def test_the_device_reports_a_plausible_size(self, accelerator):
        """40 GB on a Polaris A100, 64 GB on an Aurora tile."""
        assert accelerator.total_memory_gb() > 1.0
        assert accelerator.device_count() >= 1
