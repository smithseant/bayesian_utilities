import pytest

from numpy import array, full, concatenate, floor, isclose, log
from numpy.random import default_rng

from bayesian_utilities.space_filling import (lhs_design, sobol_design,
                                              greedy_maximin_design, support_points_design)


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