import pytest

from numpy import array, full, concatenate, floor, isclose, log, unique, arange
from numpy.random import default_rng
import pandas as pd

from bayesian_utilities.space_filling import (lhs_design, sobol_design,
                                              greedy_maximin_design, support_points_design,
                                              spacefill_permute_design, combine_spacefill_design)


# Generative space-filling designs on a unit box:

def test_lhs_one_point_per_stratum():
    n_pnts, n_dims = 8, 3
    design = lhs_design(n_pnts, n_dims, rng=default_rng(45))
    assert design.shape == (n_pnts, n_dims)
    for col in design.T:
        assert len(set(floor(col * n_pnts).astype(int))) == n_pnts


def test_lhs_ongrid_augmentation_exactly_completes_hypercube():
    n_init, n_adtl, n_dims = 5, 15, 2
    n_tot = n_init + n_adtl
    δx = 1 / n_tot
    init_indices = array([0, 4, 8, 12, 16], dtype='uint')
    init = (init_indices[:, None] + 0.5) * δx * full((1, n_dims), 1.0)
    adtl = lhs_design(n_adtl, n_dims, fixed=init, rng=default_rng(45))
    aug = concatenate((init, adtl), axis=0)
    for col in aug.T:
        # exact completion: every stratum filled exactly once...
        assert len(set(floor(col * n_tot).astype(int))) == n_tot
    all_strata = set(range(n_tot))
    fixed_strata = set(int(i) for i in init_indices)
    for col_adtl in adtl.T:
        adtl_strata = set(floor(col_adtl * n_tot).astype(int))
        # new points fill *exactly* the strata that the fixed points did not (set complement)...
        assert adtl_strata == all_strata - fixed_strata


def test_lhs_offgrid_augmentation_exactly_completes_hypercube():
    rng = default_rng(45)
    n_init, n_adtl, n_dims = 6, 14, 2
    n_tot = n_init + n_adtl
    init = lhs_design(n_init, n_dims, rng=rng)
    adtl = lhs_design(n_adtl, n_dims, fixed=init, rng=rng)
    aug = concatenate((init, adtl), axis=0)
    for col in aug.T:
        assert len(set(floor(col * n_tot).astype(int))) == n_tot


def test_lhs_colliding_augmentation_spills_gracefully():
    n_init, n_adtl, n_dims = 4, 8, 2
    n_tot = n_init + n_adtl
    # Deliberately place two initial points within the same final stratum (forcing a spill over).
    δx = 1 / n_tot
    init = array([[1.1 * δx, 4.1 * δx],  # ┐ in the 2nd dim, these collide within the 5th stratum
                  [4.5 * δx, 4.5 * δx],  # ┘
                  [7.5 * δx, 7.5 * δx],
                  [10.5 * δx, 10.5 * δx],])
    adtl = lhs_design(n_adtl, n_dims, fixed=init, rng=default_rng(45))
    assert adtl.shape == (n_adtl, n_dims)
    for col_init, col_adtl in zip(init.T, adtl.T):
        init_strata = floor(col_init * n_tot).astype(int)
        adtl_strata = floor(col_adtl * n_tot).astype(int)
        assert len(set(adtl_strata)) == n_adtl
        assert set(adtl_strata).isdisjoint(set(init_strata))


def test_sobol_requires_power_of_two():
    n_pnts, n_dims = 32, 2
    design = sobol_design(n_pnts, n_dims, rng=default_rng(45))
    assert design.shape == (32, 2)


# Sub-sampling designs for arbitrary densities:

def test_greedy_maximin_count_and_membership():
    n_prop, n_pnts, n_dims = 500, 20, 3
    proposals = default_rng(46).random((n_prop, n_dims))
    indices = greedy_maximin_design(n_pnts, proposals, return_by_index=True, rng=default_rng(45))
    assert len(indices) == n_pnts and len(set(indices)) == n_pnts


def test_support_points_weights_sum_to_n():
    n_prop, n_pnts, n_dims = 500, 20, 3
    proposals = default_rng(46).random((n_prop, n_dims))
    _, w = support_points_design(n_pnts, proposals, return_weights=True, rng=default_rng(45))
    assert isclose(w.sum(), n_pnts)


def test_fixed_exclusion_from_selection():
    n_prop, n_init, n_adtl, n_dims = 500, 5, 10, 2
    proposals = default_rng(46).random((n_prop, n_dims))
    init = default_rng(47).random((n_init, n_dims))
    indices = greedy_maximin_design(n_adtl, proposals, fixed=init, return_by_index=True,
                                    rng=default_rng(45))
    assert (len(indices)) == n_adtl


def test_greedy_maximin_requires_fixed_ln_pdfs():
    """With `ln_pdfs` & `fixed` if no `fixed_ln_pdfs`, should raise an error."""
    n_prop, n_init, n_adtl, n_dims = 500, 5, 10, 2
    proposals = default_rng(46).random((n_prop, n_dims))
    ln_pdfs = log(default_rng(48).random(n_prop) + 1e-6)
    init = default_rng(47).random((n_init, n_dims))
    with pytest.raises(ValueError):
        greedy_maximin_design(n_adtl, proposals, ln_pdfs=ln_pdfs, fixed=init, beta=1.0,
                              return_by_index=True, rng=default_rng(45))


def test_lnpdfs_and_fixed():
    """
    Greedy maximin with PDF values, fixed points, and beta > 0 will run the full algorithm and
    return `n_adtl` unique new-point indices.
    """
    n_prop, n_init, n_adtl, n_dims = 500, 5, 10, 2
    proposals = default_rng(46).random((n_prop, n_dims))
    ln_pdfs = log(default_rng(47).random(n_prop) + 1e-6)
    init = default_rng(48).random((n_init, n_dims))
    init_ln_pdfs = log(default_rng(49).random(n_init) + 1e-6)
    indices = greedy_maximin_design(n_adtl, proposals, ln_pdfs=ln_pdfs,
                                    fixed=init, fixed_ln_pdfs=init_ln_pdfs,
                                    beta=1.0, return_by_index=True, rng=default_rng(45))
    assert len(indices) == n_adtl and len(set(indices)) == n_adtl


# Combine independent groups of sampled parameters & space-fill by sub-sampling:

def _make_subsets(sizes, dims, seed=50):
    """List of ndarray subsets with given per-subset sizes & dimensionalities."""
    rng = default_rng(seed)
    return [rng.random((m, d)) for m, d in zip(sizes, dims)]


def _per_subset_distinct(design_arr, dims):
    """Count distinct rows within each subset's column block of a combined ndarray design."""
    counts, col = [], 0
    for d in dims:
        counts.append(len(unique(design_arr[:, col:col + d], axis=0)))
        col += d
    return counts


# ...for: spacefill_permute_design

def test_permute_shape_and_zero_duplication():
    D, dims = 100, [4, 6, 3]
    subsets = _make_subsets([2000, 3000, 1500], dims)
    design = spacefill_permute_design(D, subsets, rng=default_rng(45))
    assert design.shape == (D, sum(dims))
    # every subset is space-filled to D distinct points -> zero duplication by construction:
    assert _per_subset_distinct(design, dims) == [D, D, D]


def test_permute_return_by_index_list():
    D, dims = 80, [5, 5]
    subsets = _make_subsets([1000, 1000], dims)
    idx = spacefill_permute_design(D, subsets, return_by_index=True, rng=default_rng(45))
    assert isinstance(idx, list) and len(idx) == 2
    for ix, sub in zip(idx, subsets):
        assert len(ix) == D and ix.max() < sub.shape[0]


def test_permute_undersupplied_warns_and_forces_duplication():
    D, dims = 200, [4, 4]
    subsets = _make_subsets([2000, 120], dims)      # subset 1 under-supplied (120 < 200)
    with pytest.warns(UserWarning, match="forcing at least"):
        design = spacefill_permute_design(D, subsets, rng=default_rng(45))
    distinct = _per_subset_distinct(design, dims)
    assert distinct[0] == D                          # well-supplied: all distinct
    assert distinct[1] == 120                         # capped at its supply


def test_permute_dict_input_dataframe_output():
    D = 60
    rng = default_rng(51)
    subsets = {"alpha": pd.DataFrame(rng.random((800, 3)), columns=["a", "b", "c"]),
               "beta":  pd.DataFrame(rng.random((900, 2)), columns=["x", "y"])}
    design = spacefill_permute_design(D, subsets, rng=default_rng(45))
    assert isinstance(design, pd.DataFrame)
    assert len(design) == D
    # MultiIndex columns: (label, original column)
    assert list(design.columns) == [("alpha", "a"), ("alpha", "b"), ("alpha", "c"),
                                     ("beta", "x"), ("beta", "y")]


def test_permute_dict_return_by_index_is_dict():
    D = 50
    rng = default_rng(52)
    subsets = {"p": rng.random((600, 2)), "q": rng.random((600, 2))}
    idx = spacefill_permute_design(D, subsets, return_by_index=True, rng=default_rng(45))
    assert isinstance(idx, dict) and set(idx) == {"p", "q"}
    assert all(len(v) == D for v in idx.values())


# ...for: combine_spacefill_design

def test_combine_shape_and_membership():
    D, dims = 150, [5, 4, 6]
    subsets = _make_subsets([4000, 6000, 5000], dims)
    design = combine_spacefill_design(D, subsets, f_tol=0.1, rng=default_rng(45))
    assert design.shape == (D, sum(dims))


def test_combine_return_by_index_rows_align():
    D, dims = 100, [3, 3]
    subsets = _make_subsets([3000, 3000], dims)
    idx = combine_spacefill_design(D, subsets, f_tol=0.1, return_by_index=True,
                                   rng=default_rng(45))
    assert isinstance(idx, list) and all(len(ix) == D for ix in idx)
    # reconstructing from the indices reproduces the array output under the same rng seed:
    design = combine_spacefill_design(D, subsets, f_tol=0.1, rng=default_rng(45))
    rebuilt = concatenate([subsets[i][idx[i]] for i in range(len(subsets))], axis=1)
    assert isclose(rebuilt, design).all()


def test_combine_duplication_tracks_f_tol():
    """Looser f_tol -> larger R -> more duplication (fewer distinct per subset)."""
    D, dims = 400, [6, 6]
    subsets = _make_subsets([3000, 3000], dims, seed=53)
    low = combine_spacefill_design(D, subsets, f_tol=0.02, rng=default_rng(45))
    high = combine_spacefill_design(D, subsets, f_tol=0.30, rng=default_rng(45))
    dup_low = [1 - c / D for c in _per_subset_distinct(low, dims)]
    dup_high = [1 - c / D for c in _per_subset_distinct(high, dims)]
    # higher tolerance should yield at least as much duplication on the binding subsets:
    assert max(dup_high) > max(dup_low)


def test_combine_ftol_zero_uses_max_abundance():
    """In JOINT mode, f_tol=0 means 'use the max-abundance pool' (R=R_cap), NOT the
    zero-duplication endpoint — that endpoint lives in spacefill_permute_design.  So some
    within-subset duplication is expected; check the design is still valid & well-formed."""
    D, dims = 120, [4, 4]
    subsets = _make_subsets([5000, 5000], dims, seed=54)
    design = combine_spacefill_design(D, subsets, f_tol=0.0, rng=default_rng(45))
    assert design.shape == (D, sum(dims))
    # the permute variant is the actual zero-duplication route:
    permuted = spacefill_permute_design(D, subsets, rng=default_rng(45))
    assert _per_subset_distinct(permuted, dims) == [D, D]


def test_combine_undersupplied_warns_and_forces_floor():
    D, dims = 300, [4, 5]
    subsets = _make_subsets([4000, 150], dims, seed=55)   # subset 1: 150 < 300
    with pytest.warns(UserWarning, match="forcing at least"):
        design = combine_spacefill_design(D, subsets, f_tol=0.05, rng=default_rng(45))
    distinct = _per_subset_distinct(design, dims)
    # the under-supplied subset cannot exceed its own supply of distinct values:
    assert distinct[1] <= 150


def test_combine_dict_input_dataframe_output():
    D = 100
    rng = default_rng(56)
    subsets = {"m1": pd.DataFrame(rng.random((4000, 3)), columns=["u", "v", "w"]),
               "m2": pd.DataFrame(rng.random((4000, 2)), columns=["p", "q"])}
    design = combine_spacefill_design(D, subsets, f_tol=0.1, rng=default_rng(45))
    assert isinstance(design, pd.DataFrame) and len(design) == D
    assert list(design.columns) == [("m1", "u"), ("m1", "v"), ("m1", "w"),
                                     ("m2", "p"), ("m2", "q")]


def test_combine_dict_return_by_index_is_dict():
    D = 90
    rng = default_rng(57)
    subsets = {"s0": rng.random((4000, 2)), "s1": rng.random((4000, 2))}
    idx = combine_spacefill_design(D, subsets, f_tol=0.1, return_by_index=True,
                                   rng=default_rng(45))
    assert isinstance(idx, dict) and set(idx) == {"s0", "s1"}
    assert all(len(v) == D for v in idx.values())


def test_combine_max_proposals_caps_pool():
    """A small max_proposals should bound P (and thus runtime); design still produced."""
    D, dims = 100, [4, 4]
    subsets = _make_subsets([8000, 8000], dims, seed=58)
    design = combine_spacefill_design(D, subsets, f_tol=0.2, max_proposals=2000,
                                      rng=default_rng(45))
    assert design.shape == (D, sum(dims))


def test_combine_mixed_list_ndarray_and_dataframe():
    """List input mixing an ndarray and a DataFrame -> DataFrame output (positional keys)."""
    D = 70
    rng = default_rng(59)
    subsets = [rng.random((3000, 2)),
               pd.DataFrame(rng.random((3000, 3)), columns=["a", "b", "c"])]
    design = combine_spacefill_design(D, subsets, f_tol=0.1, rng=default_rng(45))
    assert isinstance(design, pd.DataFrame) and len(design) == D
    # positional top-level keys (0, 1); ndarray block gets range-integer sub-columns:
    assert design.columns[0][0] == 0 and design.columns[-1] == (1, "c")


def test_combine_tempering_pass_through_runs():
    """ln_pdfs + beta in kwargs should be accepted and produce a valid design."""
    D, dims = 100, [4, 4]
    rng = default_rng(60)
    subsets = [rng.random((3000, 4)), rng.random((3000, 4))]
    ln_pdfs = [log(rng.random(3000) + 1e-6), log(rng.random(3000) + 1e-6)]
    design = combine_spacefill_design(D, subsets, subset_ln_pdfs=ln_pdfs, f_tol=0.1,
                                      spacefill_kwargs={"beta": 0.5}, rng=default_rng(45))
    assert design.shape == (D, sum(dims))


def test_combine_support_points_method():
    """The combine machinery is method-agnostic: support_points_design should also work."""
    D, dims = 60, [3, 3]
    subsets = _make_subsets([2000, 2000], dims, seed=61)
    design = combine_spacefill_design(D, subsets, f_tol=0.1,
                                      spacefill_method=support_points_design,
                                      max_proposals=5000, rng=default_rng(45))
    assert design.shape == (D, sum(dims))


# ...the duplication model (unit-level checks on the documented properties)

def test_duplication_model_monotonicity():
    """
    Higher f_tol -> larger R -> more within-subset duplication (non-decreasing).
    (Note: f_tol=0 is NOT the zero-duplication point for JOINT mode, so it is excluded;
    the zero endpoint is spacefill_permute_design.)
    """
    D, dims = 300, [5, 5]
    subsets = _make_subsets([6000, 6000], dims, seed=62)
    dups = []
    for f_tol in (0.02, 0.08, 0.15, 0.30):
        design = combine_spacefill_design(D, subsets, f_tol=f_tol, rng=default_rng(45))
        dups.append(max(1 - c / D for c in _per_subset_distinct(design, dims)))
    assert dups == sorted(dups)          # non-decreasing in f_tol