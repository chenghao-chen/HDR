"""
Process setup — hdr_platform/runtime.py, and the doctor diagnostic.

Two things here are load-bearing on ALCF hardware in a way that is invisible
anywhere else:

  * The BLAS thread cap. Unset, OpenBLAS spawns one thread per reported core
    — 256 on a Polaris login node, 208 on an Aurora node — hits RLIMIT_NPROC
    and dies at ``import numpy`` with "blas_thread_init: pthread_create
    failed". That is a crash, not a warning, and it happens before any of
    this project's code runs.

  * Per-worker seeding. torch gives each DataLoader worker a distinct
    ``initial_seed()`` but does not propagate it to ``random``, which is what
    the D4 augmentation actually draws from. Without the propagation every
    worker applies the same flips.

The doctor tests care about one property above all: a check that raises must
be reported as a failure, not abort the diagnostic. A node is usually broken
in more than one way, and the run that stops at the first exception hides the
reason for the rest.
"""

import os
import random

import pytest
import torch

from hdr_platform.runtime import (
    RuntimeInfo, cap_blas_threads, configure, make_generator, seed_everything,
    worker_init_fn,
)


# ═════════════════════════════════════════════════════════════════════════════
# Thread caps
# ═════════════════════════════════════════════════════════════════════════════
class TestThreadCaps:
    ALL = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

    def test_every_blas_backend_is_capped(self, monkeypatch):
        """
        Capping OMP alone is not enough: OpenBLAS, MKL and NumExpr each read
        their own variable, and it only takes one uncapped library to exhaust
        the process limit.
        """
        for var in self.ALL:
            monkeypatch.delenv(var, raising=False)
        cap_blas_threads(4)
        assert all(os.environ[v] == "4" for v in self.ALL)

    def test_an_existing_value_survives_by_default(self, monkeypatch):
        """
        `OMP_NUM_THREADS=8 ./run_tests.sh` must mean eight. The caps are a
        floor against a crash, not a policy.
        """
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        assert cap_blas_threads(4) == 8
        assert os.environ["OMP_NUM_THREADS"] == "8"

    def test_force_overrides_an_existing_value(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        assert cap_blas_threads(2, force=True) == 2

    def test_zero_threads_is_rejected(self):
        """
        OMP_NUM_THREADS=0 is not "unlimited", it is undefined behaviour that
        some BLAS builds read as "spawn per core" — the exact crash the cap
        exists to prevent.
        """
        with pytest.raises(ValueError, match="threads must be >= 1"):
            cap_blas_threads(0)


# ═════════════════════════════════════════════════════════════════════════════
# Seeding
# ═════════════════════════════════════════════════════════════════════════════
class TestSeeding:
    def test_it_seeds_torch_and_python_together(self):
        """
        The augmentation pipeline draws from `random`; the model init draws
        from torch. Seeding one and not the other makes a run reproducible in
        the numbers it reports and not in the data it trains on.
        """
        seed_everything(21)
        a_torch, a_py = torch.rand(3), random.random()
        seed_everything(21)
        assert torch.equal(a_torch, torch.rand(3))
        assert a_py == random.random()

    def test_it_returns_the_seed_for_logging(self):
        assert seed_everything(1234) == 1234

    def test_numpy_is_seeded_when_present(self):
        np = pytest.importorskip("numpy")
        seed_everything(7)
        first = np.random.rand(3).tolist()
        seed_everything(7)
        assert first == np.random.rand(3).tolist()

    def test_worker_init_gives_different_workers_different_streams(self):
        """
        Two workers with the same stream apply identical augmentations to
        different images, which silently halves the effective diversity of
        every batch.
        """
        draws = []
        for worker_seed in (1000, 2000):
            torch.manual_seed(worker_seed)
            worker_init_fn(0)
            draws.append(random.random())
        assert draws[0] != draws[1]

    def test_worker_init_is_deterministic_for_a_given_stream(self):
        """Different from each other, but reproducible run to run."""
        def draw():
            torch.manual_seed(4242)
            worker_init_fn(3)
            return random.random()
        assert draw() == draw()

    def test_make_generator_is_reproducible(self):
        a = torch.randperm(10, generator=make_generator(5))
        b = torch.randperm(10, generator=make_generator(5))
        assert torch.equal(a, b)


# ═════════════════════════════════════════════════════════════════════════════
# configure()
# ═════════════════════════════════════════════════════════════════════════════
class TestConfigure:
    def test_it_returns_a_usable_device_and_seeds_everything(self):
        rt = configure(seed=21, device="cpu")
        assert isinstance(rt, RuntimeInfo)
        assert rt.device == torch.device("cpu")
        assert rt.seed == 21
        torch.zeros(1, device=rt.device)

    def test_it_is_idempotent(self):
        """
        Called at the top of the training script, the eval script and the
        smoke script, any of which may import another. A second call must not
        change the device or re-randomise anything.
        """
        a = configure(seed=21, device="cpu")
        b = configure(seed=21, device="cpu")
        assert a.device == b.device and a.seed == b.seed

    def test_seed_none_skips_seeding(self):
        """For a run that deliberately wants fresh randomness."""
        assert configure(seed=None, device="cpu").seed == -1

    def test_an_explicit_thread_count_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "16")
        assert configure(seed=1, device="cpu", threads=3).threads == 3
        assert torch.get_num_threads() == 3

    def test_the_banner_names_the_site_and_the_device(self):
        """
        This is the first thing in every job log, and the thing you read to
        find out whether the job got a GPU at all.
        """
        text = configure(seed=21, device="cpu").banner()
        assert "site" in text and "device" in text and "seed" in text

    def test_fast_matmul_can_be_declined(self):
        """
        Reduced-precision matmul changes numerics; a test comparing against a
        reference implementation needs to be able to turn it off.
        """
        configure(seed=1, device="cpu", fast_matmul=False)   # must not raise


# ═════════════════════════════════════════════════════════════════════════════
# doctor
# ═════════════════════════════════════════════════════════════════════════════
class TestDoctor:
    def test_a_full_run_produces_a_verdict_for_every_check(self):
        from hdr_platform.doctor import run_checks
        results = run_checks()
        assert len(results) >= 8
        assert all(c.status in ("ok", "warn", "fail") for c in results)
        assert all(c.name and c.detail for c in results)

    def test_it_reports_the_backends_this_host_really_has(self):
        from hdr_platform.doctor import run_checks
        by_name = {c.name: c for c in run_checks(only=["torch", "python"])}
        assert by_name["torch"].status == "ok"
        assert by_name["python"].status == "ok"

    def test_a_check_that_raises_is_a_failure_not_a_crash(self, monkeypatch):
        """
        A broken node usually fails several checks at once. Aborting on the
        first exception hides every subsequent reason, which is the opposite
        of what a diagnostic is for.
        """
        from hdr_platform import doctor

        def explode():
            raise RuntimeError("driver is on fire")

        monkeypatch.setattr(doctor, "_CHECKS", [explode, doctor.check_python])
        results = doctor.run_checks()
        assert len(results) == 2
        assert results[0].status == "fail"
        assert "driver is on fire" in results[0].detail
        assert results[1].status == "ok"

    def test_only_and_skip_select_checks(self):
        from hdr_platform.doctor import run_checks
        assert [c.name for c in run_checks(only=["site"])] == ["site"]
        assert "site" not in [c.name for c in run_checks(skip=["site"])]

    def test_exit_status_is_zero_when_nothing_failed(self, monkeypatch, capsys):
        from hdr_platform import doctor
        monkeypatch.setattr(doctor, "_CHECKS", [doctor.check_python])
        assert doctor.main([]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_a_failure_gives_a_non_zero_exit(self, monkeypatch, capsys):
        """
        The submit scripts branch on this: a failing preflight stops an
        eight-hour job before it starts.
        """
        from hdr_platform import doctor
        monkeypatch.setattr(
            doctor, "_CHECKS",
            [lambda: doctor.Check("thing", doctor.FAIL, "broken")])
        assert doctor.main([]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_strict_turns_warnings_into_a_non_zero_exit(self, monkeypatch):
        """
        The smoke job runs --strict: on a node that is about to run a real
        workload, "W&B will be offline" is worth a distinct exit code from
        "everything is fine".
        """
        from hdr_platform import doctor
        monkeypatch.setattr(
            doctor, "_CHECKS",
            [lambda: doctor.Check("thing", doctor.WARN, "iffy")])
        assert doctor.main([]) == 0
        assert doctor.main(["--strict"]) == 2

    def test_json_output_is_parseable(self, capsys):
        """Job-log parsers and the shell helpers both read this."""
        import json
        from hdr_platform import doctor
        assert doctor.main(["--json", "--only", "python"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed[0]["name"] == "python"
        assert set(parsed[0]) == {"name", "status", "detail", "hint"}

    def test_failing_checks_carry_an_actionable_hint(self, monkeypatch):
        """
        A diagnostic that says "dataset: missing" and stops has moved the
        problem, not helped. Every failure should name the fix.
        """
        from hdr_platform import doctor
        monkeypatch.setenv("HDR_DATASET_DIR", "/nonexistent/path/xyz")
        check = doctor.check_dataset()
        assert check.status == "fail"
        assert "stage_data" in check.hint or "HDR_DATASET_DIR" in check.hint

    def test_the_thread_check_notices_an_uncapped_host(self, monkeypatch):
        from hdr_platform import doctor
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert doctor.check_threads().status == "warn"

    def test_the_network_check_is_quiet_when_wandb_is_offline(self, monkeypatch):
        """
        Offline is a legitimate choice, not a degraded state; warning about a
        missing proxy there is noise at the top of every log.
        """
        from hdr_platform import doctor
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("HDR_SITE", "polaris")
        assert doctor.check_network().status == "ok"
