"""
D4 augmentation for packed BGGR Bayer — `make_d4_transform`.

Why this file matters
─────────────────────
`make_d4_transform` is the only place in the training pipeline that reorders
Bayer channels. A packed BGGR tensor [4, h, w] is not an ordinary image: each
spatial flip moves every pixel to a cell position with a *different* colour
filter, so the spatial op MUST be paired with a channel permutation that puts
the colours back where they belong. Get the permutation wrong and every
training sample is silently colour-scrambled: shapes stay right, losses still
go down, nothing crashes — the model just learns the wrong demosaicing prior.

So the ground truth used here is never the packed tensor itself; it is the
SENSOR MOSAIC. With  S(p) = F.pixel_shuffle(p, 2)  the contract is

    S(hflip(p)[[1, 0, 3, 2]])           == hflip(S(p))
    S(vflip(p)[[2, 3, 0, 1]])           == vflip(S(p))
    S(p.permute(0,2,1)[[0, 2, 1, 3]])   == S(p).transpose(-2, -1)   (square only)

i.e. "transform the packed tensor and permute" must be *exactly* the same
picture as "transform the raw sensor readout". These are tested as bit-exact
tensor equalities on hand-built mosaics with unique per-pixel values, so any
one-cell mis-permutation is caught and its offsets reported.

The transform's contract, as used by MobileHDRDataset.__getitem__:
  * it is applied PER SAMPLE, on unbatched [4, H, W] tensors;
  * one random draw is shared by the (noisy, clean) pair — they must stay
    pixel-aligned or the supervision target no longer matches the input;
  * `allow_transpose=False` (Phase 2, non-square full frames) must never
    swap H and W, otherwise collate_pad_to_max sees inconsistent aspect
    ratios.
"""

import random

import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from helpers import mosaic_from_packed
from train_A100_MoE_two_phase import make_d4_transform


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

# The permutations the module docstring promises, as *gather* index lists:
#   q = p[PERM]  means  q[c] = p[PERM[c]]
PERM_H = [1, 0, 3, 2]      # B<->G1, G2<->R
PERM_V = [2, 3, 0, 1]      # B<->G2, G1<->R
PERM_T = [0, 2, 1, 3]      # G1<->G2
PERM_180 = [3, 2, 1, 0]    # B<->R,  G1<->G2

CHANNEL_NAMES = ("B", "G1", "G2", "R")


def unique_packed(h=6, w=8):
    """
    Packed BGGR [4, h, w] whose every element is a distinct value.

    Uniqueness is what makes the mosaic identities discriminating: with
    arbitrary values a wrong permutation can coincidentally match, with
    distinct values it never can.
    """
    return torch.arange(4 * h * w, dtype=torch.float32).reshape(4, h, w)


def mosaic(p):
    """Unbatched packed [4, h, w] -> 2-D sensor mosaic [2h, 2w]."""
    return mosaic_from_packed(p.unsqueeze(0))[0, 0]


def mosaic_apply(m, bits):
    """
    Apply the D4 word described by `bits` = (hflip, vflip, transpose) to a
    2-D mosaic, in the same order make_d4_transform applies it to the packed
    tensor. Uses raw torch ops so the expectation does not depend on
    torchvision behaving the way we assume.
    """
    do_h, do_v, do_t = bits
    if do_h:
        m = m.flip(-1)
    if do_v:
        m = m.flip(-2)
    if do_t:
        m = m.transpose(-2, -1)
    return m


def draw_bits(seed, allow_transpose):
    """
    Replay the exact random draws make_d4_transform will consume for `seed`.

    The transform calls random.random() twice, and a third time only when
    allow_transpose is True (Python's `and` short-circuits), so the number of
    draws to replay depends on the flag.
    """
    random.seed(seed)
    n_draws = 3 if allow_transpose else 2
    bits = tuple(random.random() > 0.5 for _ in range(n_draws))
    return bits if allow_transpose else bits + (False,)


def seed_for_bits(allow_transpose, limit=200):
    """{(h, v, t) -> smallest seed producing that draw}."""
    found = {}
    for s in range(limit):
        found.setdefault(draw_bits(s, allow_transpose), s)
    return found


def run_transform(transform, seed, *tensors):
    """Seed python's RNG, then push `tensors` through one transform call."""
    random.seed(seed)
    return transform(*tensors)


def assert_mosaic_equal(got, want, label):
    """
    Bit-exact mosaic comparison that names the offending sensor offsets.

    The intra-cell offset (y % 2, x % 2) is reported because that is what
    identifies a mis-permutation: a wrong channel index shows up as every
    mismatch sharing one or two intra-cell parities.
    """
    assert got.shape == want.shape, \
        f"{label}: shape {tuple(got.shape)} != expected {tuple(want.shape)}"
    if torch.equal(got, want):
        return
    bad = (got != want).nonzero()
    lines = []
    for idx in bad[:8].tolist():
        y, x = idx
        lines.append(
            f"  mosaic[{y},{x}] (cell {y // 2},{x // 2} offset {y % 2},{x % 2}): "
            f"got {float(got[y, x])} want {float(want[y, x])}"
        )
    raise AssertionError(
        f"{label}: {bad.shape[0]}/{got.numel()} sensor pixels wrong\n"
        + "\n".join(lines)
    )


def compose_gathers(outer, inner):
    """
    Composition of two gather permutations.

    If q = p[outer] and r = q[inner] then r[c] = p[outer[inner[c]]], so the
    permutation equivalent to "apply outer, then apply inner" is this list.
    """
    return [outer[i] for i in inner]


# ─────────────────────────────────────────────────────────────────────────────
# 1. The three generator identities — the heart of the file
# ─────────────────────────────────────────────────────────────────────────────

def test_hflip_with_permutation_equals_mosaic_hflip():
    """
    S(hflip(p)[[1,0,3,2]]) == hflip(S(p)), bit-exact.

    Flipping columns swaps even and odd sensor columns, so the left/right
    partner of each colour changes: B<->G1 and G2<->R. If the permutation
    were dropped or wrong, the packed tensor would no longer be a BGGR
    packing of the flipped sensor image.
    """
    p = unique_packed(6, 8)
    got = mosaic(TF.hflip(p)[PERM_H])
    assert_mosaic_equal(got, mosaic(p).flip(-1), "hflip + [1,0,3,2]")


def test_vflip_with_permutation_equals_mosaic_vflip():
    """
    S(vflip(p)[[2,3,0,1]]) == vflip(S(p)), bit-exact.

    Flipping rows swaps even and odd sensor rows: B<->G2, G1<->R. Pins the
    vertical half of the Bayer-phase algebra.
    """
    p = unique_packed(6, 8)
    got = mosaic(TF.vflip(p)[PERM_V])
    assert_mosaic_equal(got, mosaic(p).flip(-2), "vflip + [2,3,0,1]")


def test_transpose_with_permutation_equals_mosaic_transpose():
    """
    S(p.permute(0,2,1)[[0,2,1,3]]) == S(p).transpose(-2,-1) on square input.

    Transposing maps (r,c)->(c,r), which fixes the two diagonal cell corners
    (B, R) and swaps the off-diagonal greens. Square-only because the packed
    spatial dims are swapped.
    """
    p = unique_packed(8, 8)
    got = mosaic(p.permute(0, 2, 1)[PERM_T].contiguous())
    assert_mosaic_equal(got, mosaic(p).transpose(-2, -1), "transpose + [0,2,1,3]")


def test_rot180_with_composed_permutation_equals_mosaic_rot180():
    """
    hflip then vflip, with the docstring's composed permutation [3,2,1,0],
    equals a 180-degree rotation of the sensor mosaic.

    This is the composite the transform actually produces when both coin
    flips come up heads, so it must hold as an identity in its own right.
    """
    p = unique_packed(6, 8)
    got = mosaic(TF.vflip(TF.hflip(p))[PERM_180])
    want = mosaic(p).flip(-1).flip(-2)
    assert_mosaic_equal(got, want, "rot180 + [3,2,1,0]")


@pytest.mark.parametrize(
    "op,correct_perm,label",
    [
        (lambda t: TF.hflip(t), PERM_H, "hflip"),
        (lambda t: TF.vflip(t), PERM_V, "vflip"),
        (lambda t: t.permute(0, 2, 1).contiguous(), PERM_T, "transpose"),
    ],
)
def test_channel_permutations_are_load_bearing(op, correct_perm, label):
    """
    Every permutation OTHER than the documented one breaks the mosaic
    identity for its spatial op.

    Without this control, the identity tests above could pass for a trivial
    reason (e.g. a symmetric input). Here we prove that exactly one of the
    24 permutations of 4 channels works, so the identity tests genuinely
    pin the Bayer phase algebra.
    """
    import itertools

    p = unique_packed(8, 8)
    ref = mosaic_apply(
        mosaic(p),
        (label == "hflip", label == "vflip", label == "transpose"),
    )
    spatial = op(p)

    n_ok = 0
    for perm in itertools.permutations(range(4)):
        if torch.equal(mosaic(spatial[list(perm)]), ref):
            n_ok += 1
            assert list(perm) == correct_perm, (
                f"{label}: permutation {list(perm)} also satisfies the mosaic "
                f"identity but the module uses {correct_perm}"
            )
    assert n_ok == 1, f"{label}: {n_ok} permutations satisfy the identity, expected 1"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pure permutation algebra (no tensors involved)
# ─────────────────────────────────────────────────────────────────────────────

def test_hflip_then_vflip_permutation_composes_to_rot180():
    """
    Composing the H-flip and V-flip gather permutations yields [3,2,1,0],
    exactly as the module docstring claims for the 180-degree rotation.

    Checked as index algebra so a docstring/implementation drift is caught
    even if no tensor test happens to exercise that corner of D4.
    """
    assert compose_gathers(PERM_H, PERM_V) == PERM_180
    assert compose_gathers(PERM_V, PERM_H) == PERM_180  # H and V commute
    # And the pairing the docstring spells out in colour terms.
    assert [CHANNEL_NAMES[i] for i in PERM_180] == ["R", "G2", "G1", "B"]


@pytest.mark.parametrize("perm,label", [(PERM_H, "hflip"), (PERM_V, "vflip"),
                                        (PERM_T, "transpose"), (PERM_180, "rot180")])
def test_channel_permutations_are_involutions(perm, label):
    """
    Each documented permutation is its own inverse.

    Every generator of D4 used here is an involution spatially (flip twice /
    transpose twice = identity), so the paired channel permutation must be
    an involution too, or a double application would leave the channels
    rotated while the pixels came home.
    """
    assert compose_gathers(perm, perm) == [0, 1, 2, 3], \
        f"{label} permutation {perm} is not an involution"


def test_generated_group_has_exactly_eight_elements():
    """
    The three generators (with their permutations) generate a group of order
    8 acting on packed tensors, i.e. the eight words produced by the
    transform's three independent coin flips are eight *distinct* symmetries.

    If two words collapsed to the same element, the augmentation would not
    sample D4 uniformly — some orientations would be twice as likely.
    """
    p = unique_packed(8, 8)
    seen = {}
    for bits in [(h, v, t) for h in (False, True)
                 for v in (False, True) for t in (False, True)]:
        out = p
        if bits[0]:
            out = TF.hflip(out)[PERM_H]
        if bits[1]:
            out = TF.vflip(out)[PERM_V]
        if bits[2]:
            out = out.permute(0, 2, 1)[PERM_T].contiguous()
        key = tuple(out.flatten().tolist())
        assert key not in seen, f"D4 words {bits} and {seen[key]} collapse to the same output"
        seen[key] = bits
    assert len(seen) == 8


# ─────────────────────────────────────────────────────────────────────────────
# 3. make_d4_transform end-to-end, driven deterministically
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("allow_transpose", [True, False])
def test_transform_output_matches_mosaic_transform_for_every_draw(allow_transpose):
    """
    For every random draw, S(transform(p)) equals the same D4 word applied
    directly to the sensor mosaic S(p).

    This is the whole contract in one test: the packed-domain implementation
    (spatial op + channel permutation, composed in the transform's own order)
    is proven equivalent to transforming the raw sensor readout. Driven by
    seeding random so each of the 4 or 8 draws is exercised exactly.
    """
    tf = make_d4_transform(allow_transpose=allow_transpose)
    p = unique_packed(8, 8)          # square: transpose is legal
    base = mosaic(p)

    seeds = seed_for_bits(allow_transpose)
    assert len(seeds) == (8 if allow_transpose else 4)

    for bits, seed in sorted(seeds.items()):
        out_n, out_c = run_transform(tf, seed, p.clone(), p.clone())
        want = mosaic_apply(base, bits)
        assert_mosaic_equal(mosaic(out_n), want, f"noisy draw {bits} (seed {seed})")
        assert_mosaic_equal(mosaic(out_c), want, f"clean draw {bits} (seed {seed})")


def test_all_eight_d4_symmetries_are_reachable():
    """
    allow_transpose=True actually reaches all 8 D4 orientations across seeds.

    Phase 1 relies on this for orientation-invariant training; if one of the
    three coin flips were dead (e.g. a `>= 1.0` threshold) the model would
    only ever see a subgroup, and this enumeration is what notices.
    """
    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(8, 8)

    outputs = {}
    for seed in range(200):
        out, _ = run_transform(tf, seed, p.clone(), p.clone())
        outputs.setdefault(tuple(out.flatten().tolist()), draw_bits(seed, True))
        if len(outputs) == 8:
            break

    assert len(outputs) == 8, (
        f"only {len(outputs)} of 8 D4 symmetries reachable in 200 seeds; "
        f"observed draws {sorted(outputs.values())}"
    )
    # Every reachable output must be a genuine D4 image of the input mosaic.
    base = mosaic(p)
    legal = {tuple(mosaic_apply(base, b).flatten().tolist())
             for b in [(h, v, t) for h in (False, True)
                       for v in (False, True) for t in (False, True)]}
    for key in outputs:
        m = mosaic(torch.tensor(key, dtype=torch.float32).reshape(4, 8, 8))
        assert tuple(m.flatten().tolist()) in legal, "output is not a D4 image of the input"


def test_allow_transpose_false_yields_exactly_four_symmetries():
    """
    Phase 2's transform samples only the 4 dimension-preserving elements of
    D4 (identity, H, V, 180) — never the transpose coset.

    A transposed sample in Phase 2 would reach collate_pad_to_max with H and
    W swapped relative to its neighbours, so this restriction is functional,
    not cosmetic.
    """
    tf = make_d4_transform(allow_transpose=False)
    p = unique_packed(8, 8)
    base = mosaic(p)

    dim_preserving = {tuple(mosaic_apply(base, (h, v, False)).flatten().tolist())
                      for h in (False, True) for v in (False, True)}
    transposing = {tuple(mosaic_apply(base, (h, v, True)).flatten().tolist())
                   for h in (False, True) for v in (False, True)}

    seen = set()
    for seed in range(200):
        out, _ = run_transform(tf, seed, p.clone(), p.clone())
        key = tuple(mosaic(out).flatten().tolist())
        assert key not in transposing, f"seed {seed} transposed despite allow_transpose=False"
        assert key in dim_preserving, f"seed {seed} produced a non-D4 output"
        seen.add(key)
    assert seen == dim_preserving, "not all 4 dimension-preserving symmetries reachable"


def test_allow_transpose_false_never_changes_shape_on_non_square():
    """
    On a NON-SQUARE input (the real Phase 2 case, full frames) the output
    shape always equals the input shape, across many seeds.

    This is the crash guard: transposing a non-square packed tensor here
    would either blow up the batch collate or corrupt padding downstream.
    """
    tf = make_d4_transform(allow_transpose=False)
    p = unique_packed(6, 10)                  # deliberately H != W
    clean = unique_packed(6, 10) + 1000.0

    for seed in range(150):
        out_n, out_c = run_transform(tf, seed, p.clone(), clean.clone())
        assert out_n.shape == p.shape, f"seed {seed}: noisy {tuple(out_n.shape)} != (4,6,10)"
        assert out_c.shape == clean.shape, f"seed {seed}: clean {tuple(out_c.shape)} != (4,6,10)"


def test_allow_transpose_true_does_swap_dims_on_non_square():
    """
    The flip side: with allow_transpose=True a non-square input *does* come
    back with H and W swapped for some draws.

    This documents exactly why Phase 2 must pass allow_transpose=False —
    the guard above is guarding against a real, reachable behaviour rather
    than a hypothetical one.
    """
    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(6, 10)

    swapped = [s for s in range(64)
               if run_transform(tf, s, p.clone(), p.clone())[0].shape == (4, 10, 6)]
    assert swapped, "allow_transpose=True never transposed a non-square input"
    for s in swapped:
        assert draw_bits(s, True)[2] is True, \
            f"seed {s} swapped dims without the transpose coin flip"


def test_square_shape_is_preserved_for_every_draw():
    """
    Square inputs keep their shape under all 8 symmetries (Phase 1 uses
    fixed-size crops and collate_xy, which stacks without padding).
    """
    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(8, 8)
    for seed in range(64):
        out_n, out_c = run_transform(tf, seed, p.clone(), p.clone())
        assert out_n.shape == (4, 8, 8) and out_c.shape == (4, 8, 8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. noisy / clean must stay in correspondence
# ─────────────────────────────────────────────────────────────────────────────

def test_identical_inputs_stay_identical_through_one_call():
    """
    A tensor and a copy of itself pushed through ONE transform call come out
    equal, for every draw.

    If the noisy and clean branches ever consumed separate random draws, the
    two outputs would diverge here — and training would regress noisy pixels
    onto a differently-oriented target.
    """
    for allow_transpose in (True, False):
        tf = make_d4_transform(allow_transpose=allow_transpose)
        p = unique_packed(8, 8)
        for seed in range(64):
            out_n, out_c = run_transform(tf, seed, p.clone(), p.clone())
            assert torch.equal(out_n, out_c), (
                f"allow_transpose={allow_transpose} seed {seed}: noisy and clean "
                "took different random draws"
            )


def test_noisy_clean_pair_stays_pixel_aligned():
    """
    A known noisy/clean pair (clean = noisy + per-pixel offset) stays aligned:
    the difference field is itself transformed, never scrambled.

    Stronger than "both outputs are some D4 image": it proves the SAME group
    element is applied to both, pixel for pixel, so supervision stays valid.
    """
    tf = make_d4_transform(allow_transpose=True)
    noisy = unique_packed(8, 8)
    offset = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8) * 0.5 + 3.0
    clean = noisy + offset

    for seed in range(64):
        out_n, out_c = run_transform(tf, seed, noisy.clone(), clean.clone())
        bits = draw_bits(seed, True)
        want_diff = mosaic_apply(mosaic(offset), bits)
        assert_mosaic_equal(mosaic(out_c) - mosaic(out_n), want_diff,
                            f"noisy/clean offset field, draw {bits} (seed {seed})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Involution / value-preservation / purity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [(True, False, False), (False, True, False),
                                  (False, False, True), (True, True, False),
                                  (True, True, True)])
def test_applying_the_same_draw_twice_is_the_identity(bits):
    """
    Every element of the transform's generating set is an involution: feeding
    the output back through the SAME draw restores the original tensor
    exactly — same pixels AND same channel order.

    Note (True, True, True) is included because hflip.vflip.transpose is also
    an involution in D4; a channel permutation that was right spatially but
    wrong in order would show up as a non-identity round trip.
    """
    seeds = seed_for_bits(True)
    if bits not in seeds:
        pytest.skip(f"draw {bits} not produced by the first 200 seeds")
    seed = seeds[bits]

    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(8, 8)
    once_n, once_c = run_transform(tf, seed, p.clone(), p.clone())
    twice_n, twice_c = run_transform(tf, seed, once_n, once_c)

    assert torch.equal(twice_n, p), f"draw {bits} applied twice is not the identity (noisy)"
    assert torch.equal(twice_c, p), f"draw {bits} applied twice is not the identity (clean)"


@pytest.mark.parametrize("allow_transpose", [True, False])
def test_every_d4_element_preserves_the_multiset_of_values(allow_transpose):
    """
    D4 elements are permutations of sensor sites: no value is created,
    dropped or duplicated.

    A bug that indexed with repeats (e.g. [1,0,3,3]) would keep the shape and
    still look plausible, but would delete one colour plane and duplicate
    another — this multiset check is what catches it.
    """
    tf = make_d4_transform(allow_transpose=allow_transpose)
    p = torch.rand(4, 8, 8, generator=torch.Generator().manual_seed(7))
    want = torch.sort(p.flatten()).values

    for seed in range(64):
        out, _ = run_transform(tf, seed, p.clone(), p.clone())
        assert out.numel() == p.numel()
        got = torch.sort(out.flatten()).values
        assert torch.equal(got, want), (
            f"allow_transpose={allow_transpose} seed {seed}: value multiset changed "
            f"(draw {draw_bits(seed, allow_transpose)})"
        )
        # ... and each channel plane keeps its own multiset, just relabelled.
        planes_in = sorted(tuple(torch.sort(c.flatten()).values.tolist()) for c in p)
        planes_out = sorted(tuple(torch.sort(c.flatten()).values.tolist()) for c in out)
        assert planes_in == planes_out, "a channel plane was duplicated or dropped"


def test_transform_does_not_mutate_its_inputs():
    """
    The transform is pure: the caller's tensors are untouched.

    MobileHDRDataset hands it tensors that may be views of an mmap'd file
    (torch.load(..., mmap=True) plus a crop), so an in-place flip would
    corrupt the on-disk-backed page cache for every later epoch.
    """
    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(8, 8)
    clean = p + 500.0
    p_ref, clean_ref = p.clone(), clean.clone()

    for seed in range(64):
        tf_out = run_transform(tf, seed, p, clean)
        assert torch.equal(p, p_ref), f"seed {seed}: noisy input mutated in place"
        assert torch.equal(clean, clean_ref), f"seed {seed}: clean input mutated in place"
        del tf_out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dtype_preserved_and_output_is_pixel_shuffle_ready(dtype):
    """
    Output keeps its dtype and is laid out so F.pixel_shuffle works on it.

    The training loop feeds these straight into collate + PixelUnshuffle
    stages; a dtype promotion would break AMP bookkeeping and a stale
    non-contiguous view would silently cost a copy per sample.
    """
    tf = make_d4_transform(allow_transpose=True)
    p = unique_packed(8, 8).to(dtype)
    for seed in range(64):
        out, _ = run_transform(tf, seed, p.clone(), p.clone())
        assert out.dtype == dtype, f"seed {seed}: dtype {out.dtype} != {dtype}"
        assert out.is_contiguous(), f"seed {seed}: output is not contiguous"
        m = F.pixel_shuffle(out.unsqueeze(0), 2)
        assert m.shape == (1, 1, 2 * out.shape[1], 2 * out.shape[2])


def test_transform_is_per_sample_not_per_batch():
    """
    The transform indexes dim 0 as the CHANNEL axis, so it must be called on
    unbatched [4, H, W] tensors — a collated [B, 4, h, w] batch is not
    supported and does not silently pass.

    Pins the contract MobileHDRDataset.__getitem__ relies on: augmenting
    after collate would permute batch items instead of Bayer channels.
    """
    tf = make_d4_transform(allow_transpose=True)
    batched = torch.rand(2, 4, 8, 8)
    seeds = seed_for_bits(True)
    hflip_seed = seeds[(True, False, False)]

    with pytest.raises(IndexError):
        run_transform(tf, hflip_seed, batched.clone(), batched.clone())
