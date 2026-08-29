"""
Site detection and path resolution — hdr_platform/site.py.

This module decides, before anything else runs, which machine the job is on.
Getting it wrong is expensive in a specific way: a Polaris answer on Aurora
points HDR_PYTHON at an eagle path that does not exist and the job dies at
once (loud, cheap), while an Aurora answer on Polaris silently sends the run
looking for /lus/flare and fails after the queue wait (quiet, expensive).

Everything here is pure and injectable — no torch, no real filesystem — so
every branch is reachable from a login node.
"""

import os

import pytest

from hdr_platform.site import (
    AURORA, LOCAL, POLARIS, SITES, SiteSpec,
    detect_site, get_site, resolve_dataset_dir, resolve_project_root,
)


# ═════════════════════════════════════════════════════════════════════════════
# Detection precedence
# ═════════════════════════════════════════════════════════════════════════════
class TestDetection:
    def test_hdr_site_overrides_everything(self):
        """
        The explicit override has to win even when every other signal points
        elsewhere — that is what makes `HDR_SITE=aurora ./scripts/...` a
        usable way to dry-run the other machine's path from this one.
        """
        assert detect_site(hostname="polaris-login-01",
                           environ={"HDR_SITE": "aurora"}) == "aurora"

    def test_an_unknown_hdr_site_raises_rather_than_falling_back(self):
        """
        A typo must not degrade to "local". Silently treating a mistyped
        HDR_SITE as an unrecognised laptop would run an eight-hour training
        job on the CPU and report success.
        """
        with pytest.raises(ValueError, match="not a known site"):
            detect_site(hostname="polaris-login-01",
                        environ={"HDR_SITE": "polrais"})

    @pytest.mark.parametrize("host,expected", [
        ("polaris-login-01", "polaris"),
        ("POLARIS-LOGIN-02", "polaris"),
        ("aurora-uan-0009", "aurora"),
        ("uan-0012", "aurora"),
    ])
    def test_login_node_hostnames(self, host, expected):
        assert detect_site(hostname=host, environ={}) == expected

    def test_a_compute_node_hostname_alone_is_not_enough(self, monkeypatch):
        """
        Compute nodes on both machines are named like x4204c3s3b0n0, so the
        hostname genuinely cannot distinguish them. Detection must fall
        through to the filesystem check rather than guessing from the 'x'.
        """
        monkeypatch.setattr(os.path, "isdir", lambda p: p == "/lus/flare")
        assert detect_site(hostname="x4204c3s3b0n0", environ={}) == "aurora"

        monkeypatch.setattr(os.path, "isdir", lambda p: p == "/lus/eagle")
        assert detect_site(hostname="x3104c0s19b0n0", environ={}) == "polaris"

    def test_flare_wins_over_eagle_if_both_somehow_appear(self, monkeypatch):
        """
        Ordering is deliberate, not incidental: flare is checked first. If a
        host ever mounted both, Aurora is the machine whose toolchain is the
        fussy one, so that is the safer guess.
        """
        monkeypatch.setattr(os.path, "isdir",
                            lambda p: p in ("/lus/flare", "/lus/eagle"))
        assert detect_site(hostname="x0000c0s0b0n0", environ={}) == "aurora"

    def test_pbs_o_host_is_the_last_resort(self, monkeypatch):
        """
        A PBS job inherits the submitting host, which is always a login node
        and therefore always names its machine. Worth consulting, but only
        after the filesystem, which describes where the job is actually
        running rather than where it was launched from.
        """
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        assert detect_site(hostname="x4204c3s3b0n0",
                           environ={"PBS_O_HOST": "aurora-uan-0009"}) == "aurora"

    def test_an_unrecognised_host_is_local_not_an_error(self, monkeypatch):
        """A laptop or CI container must work, not raise."""
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        assert detect_site(hostname="my-laptop", environ={}) == "local"

    def test_detection_never_raises_on_a_missing_hostname(self, monkeypatch):
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        assert detect_site(hostname="", environ={}) == "local"
        assert detect_site(hostname=None, environ={}) in SITES


# ═════════════════════════════════════════════════════════════════════════════
# The specs themselves
# ═════════════════════════════════════════════════════════════════════════════
class TestSiteSpecs:
    def test_every_registered_site_is_addressable_by_name(self):
        for name, spec in SITES.items():
            assert get_site(name) is spec
            assert spec.name == name

    def test_get_site_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unknown site"):
            get_site("frontier")

    def test_the_two_machines_disagree_about_the_accelerator(self):
        """
        The single fact this whole package exists for. If these ever match,
        something has been copy-pasted.
        """
        assert POLARIS.accelerator == "cuda"
        assert AURORA.accelerator == "xpu"
        assert LOCAL.accelerator == "cpu"

    def test_aurora_tiles_are_counted_as_separate_devices(self):
        """
        Each Max 1550 is two tiles. Under the FLAT hierarchy the job sees 12
        devices of 64 GB, not 6 of 128 GB, and the memory number is the one
        that decides whether full-resolution Phase 2 fits.
        """
        assert AURORA.tiles_per_gpu == 2
        assert AURORA.devices_per_node == 12
        assert AURORA.memory_per_device_gb == 64.0

    def test_a_polaris_gpu_is_one_device(self):
        assert POLARIS.tiles_per_gpu == 1
        assert POLARIS.devices_per_node == 4
        assert POLARIS.memory_per_device_gb == POLARIS.gpu_memory_gb == 40.0

    def test_an_aurora_tile_has_more_memory_than_a_polaris_gpu(self):
        """
        The reason Phase 2 is worth attempting on Aurora at all: 64 GB per
        tile against 40 GB per A100. If this inverts, the Phase 2 guidance in
        the job scripts is wrong.
        """
        assert AURORA.memory_per_device_gb > POLARIS.memory_per_device_gb

    def test_polaris_prod_cannot_take_a_single_node_job(self):
        """
        Polaris' prod queue requires >= 10 nodes, which is exactly why the
        long single-node training job has to be preemptable — and why the
        train script is marked rerunnable with a pinned save folder.
        """
        assert POLARIS.prod_min_nodes >= 10
        assert POLARIS.queue_long == "preemptable"

    def test_aurora_does_not_need_a_preemptable_queue(self):
        """The corresponding relief on Aurora: prod accepts one node."""
        assert AURORA.prod_min_nodes == 1
        assert AURORA.queue_long == "prod"

    def test_the_filesystems_are_disjoint(self):
        """
        `#PBS -l filesystems=` must name the right one; a job that touches a
        filesystem it did not request can be killed mid-run. These two share
        only home.
        """
        polaris = set(POLARIS.filesystems.split(":"))
        aurora = set(AURORA.filesystems.split(":"))
        assert polaris & aurora == {"home"}
        assert "eagle" in polaris and "flare" in aurora

    def test_select_line_carries_the_polaris_only_system_selector(self):
        """
        `system=polaris` is a Polaris-ism. Sending it to Aurora's scheduler
        is a submission error, so the two must not share a select line.
        """
        assert POLARIS.select_line(1) == "select=1:system=polaris"
        assert AURORA.select_line(1) == "select=1"
        assert POLARIS.select_line(4) == "select=4:system=polaris"

    def test_both_machines_need_the_same_outbound_proxy(self):
        """Compute nodes have no direct route on either machine."""
        assert POLARIS.proxy == AURORA.proxy
        assert "proxy.alcf.anl.gov" in POLARIS.proxy

    def test_to_dict_is_json_friendly_and_includes_the_derived_fields(self):
        """`python -m hdr_platform.site --json` is read by shell helpers."""
        import json
        d = AURORA.to_dict()
        assert json.loads(json.dumps(d))["devices_per_node"] == 12
        assert d["memory_per_device_gb"] == 64.0

    def test_specs_are_frozen(self):
        """
        Immutable so that one caller cannot mutate the table another caller
        is about to read — these are process-global constants.
        """
        with pytest.raises(Exception):
            POLARIS.gpus_per_node = 8      # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# Paths
# ═════════════════════════════════════════════════════════════════════════════
class TestProjectRoot:
    def test_hdr_proj_root_wins_outright(self):
        env = {"HDR_PROJ_ROOT": "/somewhere/else/", "USER": "ryanchen"}
        assert resolve_project_root(POLARIS, env) == "/somewhere/else"

    def test_the_default_follows_the_machines_scratch_filesystem(self):
        env = {"USER": "ryanchen"}
        assert resolve_project_root(POLARIS, env) == \
            "/lus/eagle/projects/lighthouse-purdue/ryanchen"
        assert resolve_project_root(AURORA, env) == \
            "/lus/flare/projects/lighthouse-purdue/ryanchen"

    def test_hdr_project_renames_the_allocation(self):
        """
        The Aurora award is a separate allocation from the Polaris one and is
        frequently named differently. Without this override the Aurora paths
        are simply wrong for most users.
        """
        env = {"USER": "ryanchen", "HDR_PROJECT": "aurora-alloc"}
        assert resolve_project_root(AURORA, env) == \
            "/lus/flare/projects/aurora-alloc/ryanchen"

    def test_logname_substitutes_for_a_missing_user(self):
        """Batch environments do not always carry USER."""
        env = {"LOGNAME": "someone"}
        assert resolve_project_root(POLARIS, env).endswith("/someone")

    def test_local_falls_back_to_the_checkout(self):
        """
        No shared filesystem to build a path from, so the repository itself
        is the only sensible root.
        """
        root = resolve_project_root(LOCAL, {"USER": "x"})
        assert os.path.isdir(root)
        assert os.path.isdir(os.path.join(root, "hdr_platform"))


class TestDatasetDir:
    def test_hdr_dataset_dir_wins(self):
        """
        Every existing submit script sets this variable; the resolver must
        not quietly outvote them.
        """
        env = {"HDR_DATASET_DIR": "/data/Mobile-HDR/", "USER": "x"}
        assert resolve_dataset_dir("Mobile-HDR", POLARIS, env) == "/data/Mobile-HDR"

    def test_dataset_root_names_the_containing_directory(self):
        env = {"HDR_DATASET_ROOT": "/scratch/sets", "USER": "x"}
        assert resolve_dataset_dir("kalantari2017", AURORA, env) == \
            "/scratch/sets/kalantari2017"

    def test_the_default_lands_under_the_project_root(self):
        env = {"USER": "ryanchen"}
        assert resolve_dataset_dir("Mobile-HDR", AURORA, env) == \
            "/lus/flare/projects/lighthouse-purdue/ryanchen/datasets/Mobile-HDR"

    def test_the_same_dataset_resolves_differently_per_machine(self):
        """
        The concrete reason scripts/stage_data.sh exists: these two paths are
        on filesystems that do not see each other.
        """
        env = {"USER": "ryanchen"}
        polaris = resolve_dataset_dir("Mobile-HDR", POLARIS, env)
        aurora = resolve_dataset_dir("Mobile-HDR", AURORA, env)
        assert polaris != aurora
        assert polaris.startswith("/lus/eagle")
        assert aurora.startswith("/lus/flare")


# ═════════════════════════════════════════════════════════════════════════════
# The CLI the shell helpers call
# ═════════════════════════════════════════════════════════════════════════════
class TestCLI:
    def test_field_prints_one_bare_value(self, capsys):
        """
        Shell scripts capture this with $(...), so a stray label or blank
        line would end up inside a path.
        """
        from hdr_platform.site import _main
        assert _main(["--site", "aurora", "--field", "filesystems"]) == 0
        out = capsys.readouterr().out
        assert out == "home:flare\n"

    def test_json_round_trips(self, capsys):
        import json
        from hdr_platform.site import _main
        assert _main(["--site", "polaris", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["accelerator"] == "cuda"
        assert data["project_root"].startswith("/lus/eagle")

    def test_an_unknown_field_is_an_error_listing_the_real_ones(self, capsys):
        from hdr_platform.site import _main
        with pytest.raises(SystemExit):
            _main(["--site", "polaris", "--field", "gpu_count"])
        assert "available:" in capsys.readouterr().err

    def test_the_human_report_covers_every_field(self, capsys):
        from hdr_platform.site import _main
        assert _main(["--site", "aurora"]) == 0
        out = capsys.readouterr().out
        for key in ("accelerator", "project_root", "dataset_dir", "queue_long"):
            assert key in out
