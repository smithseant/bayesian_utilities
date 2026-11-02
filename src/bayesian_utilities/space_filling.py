"""
This is a simple toolkit to perform design-of-experiment sampling according to my preferred
methods - Latin hyper-cube sampling, Sobol sampling & greedy maximin subsampling.

For motivation, method-selection guidance, and detailed recommendations,
see `docs/space_filling.md`.

Created April 2017 @author: Sean T. Smith, updated July-Aug 2026.
"""

from warnings import warn
from math import ceil, log2
from typing import Callable, Literal, Union, Optional
from numpy import (array, full, linspace, arange, concatenate, tril_indices, unravel_index,
                   bincount, isfinite, outer, sqrt, exp, log, inf)
from numpy.linalg import svd
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.stats import qmc
import pandas as pd

# Shared type aliases:
Scale = Optional[Literal["range", "std", "cov"]]
Subsets = Union[list, dict]  # list/dict of ndarray or DataFrame items
LnPdfs = Optional[Union[list, dict]]
SpacefillMethod = Callable[..., object]


# Generative space-filling designs on a unit box:

def lhs_design(n_pnts: int, n_dims: int, fixed: Optional[NDArray] = None,
               position: Literal["center", "edges", "random"] = "center",
               rng: Optional[Generator] = None) -> NDArray:
    """
    Sample a latin hypercube design (a space-filling design which ensures a coordinate-aligned
    uniform stratification developed by Mike Mckay at Los Alamos in 1979) over the unit box w/
    `n_pnts` number of points & `n_dims` number of dimensions.  If fixed points are provided,
    their occupied strata are excluded so that the new points complete a `len(fixed) + n_pnts`
    latin hypercube (or augment an approximate one).  Points within each stratum can be placed
    at: the 'center', the 'edges', or at a 'random' location within each stratum.
    """
    rng = default_rng() if rng is None else rng
    n_fixed = 0 if fixed is None else len(fixed)
    n_tot = n_pnts + n_fixed
    δx = 1 / (n_pnts + n_fixed)
    include_endpoint = (position == "edges")
    all_strata = []
    for dim in range(n_dims):
        strata_edges = list(linspace(0, 1, n_tot, endpoint=include_endpoint))
        available_idx = list(range(n_tot))
        if fixed is not None:
            for coordinate in fixed[:, dim]:
                location_idx = min(int(coordinate * n_tot), n_tot - 1)
                # If the index of this point's location is already claimed, spill to the nearest
                # still available stratum (only approximating the completion of a full LHS).
                claimed_idx = min(range(len(available_idx)),
                                  key=lambda k: abs(available_idx[k] - location_idx))
                available_idx.pop(claimed_idx)
        available_strata = [strata_edges[i] for i in available_idx]
        all_strata.append(rng.permutation(available_strata))
    design = array(all_strata).T
    if position == "center":
        design += δx / 2
    elif position == "random":
        design += δx * rng.random((n_pnts, n_dims))
    return design


def oa_lhs_design(n_pnts: int, n_dims: int, rng: Optional[Generator] = None) -> NDArray:
    """
    Sample an orthogonal-array latin hypercube design of strength 2 (a space-filling design which
    ensures a pairwise stratified uniformity) over the unit box with `n_pnts` number of points &
    `n_dims` number of dimensions — `n_pnts` must be the square of a prime number which prime must
    be greater than or equal to `n_dims - 1`.
    Note:  for orthogonal arrays w/ strength > 2, Sobol is known to be more performant.
    (This is simply a wrapper on `scipy.stats.qmc.LatinHypercube` for the sake of a uniform calls.)
    """
    sp_lhs = qmc.LatinHypercube(d=n_dims, strength=2, rng=rng)
    return sp_lhs.random(n=n_pnts)


def sobol_design(n_pnts: int, n_dims: int, rng: Optional[Generator] = None, **kwargs) -> NDArray:
    """
    Sample a Sobol-sequence design (a space-filling design which aims for low discrepancy) over
    the unit box with `n_pnts` number of points & `n_dims` number of dimensions — `n_pnts` must
    be an integer power of 2.
    (This is simply a wrapper on `scipy.stats.qmc.Sobol` for the sake of a uniform call signature.)
    """
    sp_sobol = qmc.Sobol(d=n_dims, rng=rng, **kwargs)
    return sp_sobol.random_base2(ceil(log2(n_pnts)))


def optimal_of_designs(n_pnts: int, n_dims: int, n_samples: int,
                       sampler: Callable[..., NDArray] = lhs_design,
                       optimality: Literal["maximin", "D", "A", "I", "G"] = "maximin",
                       test_pnts: Optional[NDArray] = None,
                       pnts2bases: Optional[Callable[[NDArray], NDArray]] = None,
                       **kwargs) -> NDArray:
    """
    Optimize a space-filling property of designs by sampling `n_samples` number of designs from
    `sampler(n_pnts, n_dims, **kwargs)` over the unit box with `n_pnts` number of points and
    `n_dims` number of dimensions — and select the one with the best `optimality` property:
    
        "maximin" - maximize smallest neighbor distance (for larger closest-pair spacing),
        "D"       - maximize differential posterior Shannon info. — max(det(XᵀX))
        "A"       - minimize avg. posterior variance of the linear-parameters — min(trace((XᵀX)⁻¹)),
        "I"       - minimize the average prediction uncertainty across `test_pnts`,
        "G"       - minimize the largest prediction uncertainty among `test_pnts`.
    """

    def maximin_opt(design):
        """
        For a specified `design`, optimize by minimum inter-point distance with preference for
        fewer repeats of the same distance.
        """
        dist_array = cdist(design, design)
        dist_vals = dist_array[*tril_indices(design.shape[0], -1)]
        min_dist = dist_vals.min()
        min_count = (dist_vals == min_dist).sum()
        return min_dist, -min_count

    def d_opt(design):
        """Differential posterior Shannon info. -> determinant of the information (maximized)."""
        X = pnts2bases(design)
        U, s, VT = svd(X, full_matrices=False)
        return 2 * log(s).sum()

    def a_opt(design):
        """Avg. posterior variance of regression coefficients -> trace of inv. inf. (minimized)."""
        X = pnts2bases(design)
        U, s, VT = svd(X, full_matrices=False)
        return -(1 / s**2).sum()

    def i_opt(design):
        """Average uncertainty of posterior prediction (minimized)."""
        X = pnts2bases(design)
        U, s, VT = svd(X, full_matrices=False)
        H = pnts2bases(test_pnts) if test_pnts is not None else X
        HpinvXT = (H @ VT.T) / s
        return -((HpinvXT * HpinvXT).sum(axis=1)).mean()

    def g_opt(design):
        """Largest uncertainty of posterior prediction (minimized)."""
        X = pnts2bases(design)
        U, s, VT = svd(X, full_matrices=False)
        H = pnts2bases(test_pnts) if test_pnts is not None else X
        HpinvXT = (H @ VT.T) / s
        return -((HpinvXT * HpinvXT).sum(axis=1)).max()


    criteria = dict(maximin=maximin_opt, D=d_opt, A=a_opt, I=i_opt, G=g_opt)
    if optimality in ("D", "A", "I", "G") and pnts2bases is None:
        raise ValueError(f"optimality {optimality} requires a `pnts2bases` function")
    calc_criterion = criteria[optimality]

    opt_design = None
    opt_criterion = None
    for i in range(n_samples):
        prop_design = sampler(n_pnts, n_dims, **kwargs)
        prop_criterion = calc_criterion(prop_design)
        if opt_criterion is None or prop_criterion > opt_criterion:
            opt_design, opt_criterion = prop_design, prop_criterion
    return opt_design


# Sub-sampling designs for arbitrary densities:

def _scale_util(proposals: NDArray, fixed: Optional[NDArray],
                scale: Scale) -> tuple[NDArray, Optional[NDArray]]:
    """Scale both `proposals` and `fixed` (when not None) by: `None`, "range", "std", or "cov"."""
    fixed_scaled = None
    if scale is None:
        props_scaled = proposals
        if fixed is not None:
            fixed_scaled = fixed
    elif scale == "range":
        p_min = proposals.min(axis=0, keepdims=True)
        p_max = proposals.max(axis=0, keepdims=True)
        props_scaled = (proposals - p_min) / (p_max - p_min + 1e-12)
        if fixed is not None:
            fixed_scaled = (fixed - p_min) / (p_max - p_min + 1e-12)
    elif scale == "std":
        p_mean = proposals.mean(axis=0, keepdims=True)
        p_std = proposals.std(axis=0, keepdims=True)
        props_scaled = (proposals - p_mean) / (p_std + 1e-12)
        if fixed is not None:
            fixed_scaled = (fixed - p_mean) / (p_std + 1e-12)
    elif scale == "cov":
        p_mean = proposals.mean(axis=0, keepdims=True)
        U, s, Vt = svd(proposals - p_mean, full_matrices=False)
        use_dims = s > 1e-8 * s[0]
        props_scaled = (proposals - p_mean) @ Vt[use_dims].T / s[use_dims]
        if fixed is not None:
            fixed_scaled = (fixed - p_mean) @ Vt[use_dims].T / s[use_dims]
    else:
        raise ValueError(f"unsupported scale: {scale!r}")
    return props_scaled, fixed_scaled

def _stochastic_pick(objective: NDArray, stoch_frac: float, rng: Generator) -> int:
    """
    Pick an index from `objective` — sampling from the top `stoch_frac` fraction of candidates
    w/ a Boltzmann/softmax weighting (whose temperature scales w/ `stoch_frac`).
    If `stoch_frac == 0.0`, return the deterministic best (`argmin`).
    """
    n_all = objective.shape[0]
    if stoch_frac <= 0.0:
        return int(objective.argmin())
    n_stoch = max(int(stoch_frac * n_all), 1)
    i_best = objective.argpartition(n_stoch - 1)[:n_stoch]
    i_best = i_best[isfinite(objective[i_best])]  # (can't take `std` with `inf` values)
    if i_best.size == 1:
        return int(i_best[0])
    obj_best = objective[i_best]
    # temperature scales w/ the spread of the candidate set and w/ `stoch_frac`...
    temp = stoch_frac * (obj_best.std() + 1e-12)
    p_best = exp(-(obj_best - obj_best.min()) / temp)
    p_best /= p_best.sum()
    # Boltzmann/softmax selection
    i_selected = int(rng.choice(i_best, p=p_best))
    return i_selected


def greedy_maximin_design(n_pnts: int, proposals: NDArray, ln_pdfs: Optional[NDArray] = None,
                          fixed: Optional[NDArray] = None, fixed_ln_pdfs: Optional[NDArray] = None,
                          scale: Scale = "cov", beta: float = 0.0, stoch_frac: float = 0.1,
                          return_weights: bool = False, return_by_index: bool = False,
                          rng: Optional[Generator] = None) -> object:
    """
    Create a space-filling design w/ `n_pnts` number of output points that is selected in order by
    greedily sub-selecting points from a larger `proposals` set according to best maximin value.
    If the log values of the unscaled distribution are provided for each point, the algorithm is
    modified to density-weighted & tempered greedy maximin (with tempering parameter `beta` — where
    `beta=0.0` reduces to the unweighted algorithm, and `beta=1.0` will result in samples that are
    approximately representative of the distribution).  Existing `fixed` points can be provided —
    and if `ln_pdfs` were given then corresponding `fixed_ln_pdfs` should also be supplied (using
    the same distribution and same normalization as `ln_pdfs` for consistent spacing).
    The provided `proposals` should be scaled — options are `None`,  "range", "std" or "cov".
    The greedy optimization has an optional stochastic feature controlled by `stoch_frac`
    specifying the fraction of points considered (`stoch_frac=0.0` is fully deterministic while
    larger values up to `stoch_frac=1.0` increase the candidate pool).

    Variations go by different names, but all use the same principles & arrive at similar results:
    - The baseline additive approach is referred to by statisticians as "greedy-maximin sampling",
    - The baseline subtractive approach is referred to by computer scientists as "progressive
      Poisson-disk sampling" or "blue-noise sampling",
    Although there is a large variety of valid algorithms, one caution is taken here to avoid a
    naive implementation resulting in computation of O(n_pnts^2 * n_proposals * n_dims) in favor of
    one that gives O(n_pnts * n_proposals * n_dims).  (Mitchell 1991 demonstrated the first
    O(n_pnts * n_proposals * n_dims) algorithm, and Bridson 2007 went further by demonstrating an
    O(n_pnts * n_dims) algorithm — but Bridson seems to only work in low number of dimensions.  So,
    this implementation opts for the Mitchell approach and seems to be sufficiently fast.)
    """
    rng = default_rng() if rng is None else rng
    choice = rng.choice
    n_proposals, n_dims = proposals.shape
    props_scaled, fixed_scaled = _scale_util(proposals, fixed, scale)

    # Density weighting: higher-density & larger beta -> smaller length-scale -> farther spaced
    if ln_pdfs is None:
        l = full(n_proposals, 1.0)
    else:
        l = exp(-(beta / n_dims) * ln_pdfs)

    if fixed is None:
        # Initialize by selecting the two most distant points in a small subset of `proposals`:
        ind = choice(arange(n_proposals), max(n_pnts // 3, min(n_proposals, 20)), replace=False)
        dist_init = cdist(props_scaled[ind], props_scaled[ind]) / sqrt(outer(l[ind], l[ind]))
        i_selected = [int(arg) for arg in unravel_index(dist_init.argmax(), dist_init.shape)]
        n_new = 2
    else:
        # Initialize with the `fixed` points only:
        n_fixed = fixed.shape[0]
        props_scaled = concatenate((fixed_scaled, props_scaled), axis=0)
        if ln_pdfs is None:
            l_fixed = full(n_fixed, 1.0)
        elif fixed_ln_pdfs is None:
            raise ValueError("When both `ln_pdfs` & `fixed` are provided, `fixed_ln_pdfs` must "
                             "also be provided (evaluated from the same distribution, w/ the "
                             "same normalization as `ln_pdfs`")
        else:
            l_fixed = exp(-(beta / n_dims) * fixed_ln_pdfs)
        l = concatenate((l_fixed, l))
        n_proposals += n_fixed
        i_selected = list(range(n_fixed))
        n_new = 0

    # For each point in `proposals`, find the index of & distance to the nearest of those selected:
    i_nearest = full(n_proposals, n_proposals, dtype='uint')
    dist2nearest = full(n_proposals, inf, dtype='f8')
    for ind in i_selected:
        dist2ind = (cdist(props_scaled[[ind]], props_scaled) / sqrt(outer(l[[ind]], l))).reshape(-1)
        update_mask = dist2ind < dist2nearest
        i_nearest[update_mask] = ind
        dist2nearest[update_mask] = dist2ind[update_mask]
        # ensure each selected point is its own nearest selected point...
        i_nearest[ind] = ind
        # prevent selected points from being selected again...
        dist2nearest[ind] = -inf

    # CORE of GREEDY MAXIMIN: repeatedly select from points farthest from all previously selected
    for n_selected in range(n_new, n_pnts):
        # stochastic selection based on each candidates distance...
        ind = _stochastic_pick(-dist2nearest, stoch_frac, rng)
        i_selected.append(ind)
        # increment `i_nearest` & `dist2nearest` by comparing only to the newly selected point...
        dist2ind = (cdist(props_scaled[[ind]], props_scaled) / sqrt(outer(l[[ind]], l))).reshape(-1)
        update_mask = dist2ind < dist2nearest
        i_nearest[update_mask] = ind
        dist2nearest[update_mask] = dist2ind[update_mask]
        i_nearest[ind] = ind      # (selected point is its own nearest selected)
        dist2nearest[ind] = -inf  # (prevent selected points from being selected again)

    if return_weights:
        # Calculate weights based on the number of nearest neighbors (approx. Voronoi cell vol.):
        #  <Referred to in the optimal-quantization literature as: Wasserstein-optimal weights.>
        counts = bincount(i_nearest.astype('int64'), minlength=n_proposals)[i_selected]
        weights = (n_pnts / n_proposals) * counts
    if fixed is not None:
        # Dereference `i_selected` from the concatenated array to `proposals`:
        i_selected = i_selected[n_fixed:]
        i_selected = [el - n_fixed for el in i_selected]
        if return_weights:
            # weights for `fixed` points & weights for point selected from `proposals`...
            weights = weights[:n_fixed], weights[n_fixed:]

    # Setup the return arrays according to the input preferences:
    if return_by_index:
        ret_val = i_selected
    else:
        ret_val = proposals[i_selected]
    if return_weights:
        ret_val = (ret_val, weights)
    return ret_val


def support_points_design(n_pnts: int, proposals: NDArray, ln_pdfs: Optional[NDArray] = None,
                          fixed: Optional[NDArray] = None, scale: Scale = "cov", beta: float = 1.0,
                          stoch_frac: float = 0.1, return_weights: bool = False,
                          return_by_index: bool = False,
                          rng: Optional[Generator] = None) -> object:
    """
    Create a space-filling design w/ `n_pnts` number of output points that is selected in order by
    greedy sub-selecting points from a larger `proposals` set according to a support-points
    objective.  If the log values of the unscaled distribution are provided for each point, the
    algorithm is modified to tempered support points (with tempering parameter `beta` — where
    `beta->0.0` approaches a flat distribution — favoring the most distant points, and `beta=1.0`
    reduces to the untempered algorithm).  Existing `fixed` points can be provided.
    Caution: since the fixed points are not assumed to be representative of the distribution,
    unless `n_pnts` >> `len(fixed)` the resulting samples may also not be representative.
    The provided `proposals` should be scaled — options are `None`,  "range", "std" or "cov".
    The greedy optimization has an optional stochastic feature controlled by `stoch_frac`
    specifying the fraction of points considered (`stoch_frac=0.0` is fully deterministic while
    larger values up to `stoch_frac=1.0` increase the candidate pool).
    """
    rng = default_rng() if rng is None else rng
    choice = rng.choice
    n_proposals, n_dims = proposals.shape
    props_scaled, fixed_scaled = _scale_util(proposals, fixed, scale)

    # Density weighting: temper the distribution so lower beta -> outlying points increasing favor
    if ln_pdfs is None or beta == 1.0:
        tmp_wt = full(n_proposals, 1.0)
    else:
        tmp_wt = exp((beta - 1) * ln_pdfs - ((beta - 1) * ln_pdfs).max())
        tmp_wt /= tmp_wt.sum() / n_proposals

    # Precompute each sample's expected distance (calculated once & reused on each iteration):
    props_distances = cdist(props_scaled, props_scaled)
    expected_sample_dist = (tmp_wt @ props_distances) / n_proposals

    i_selected = []
    if fixed is None:
        # For the first point select one w/ an expected distance in the lower 50 percentile:
        n_half = max(n_proposals // 2, 1)
        i_selected.append(choice(expected_sample_dist.argpartition(n_half - 1)[:n_half]))
        n_initial, n_fixed = 1, 0
        sum_dists_to_selected = props_distances[:, i_selected].reshape(-1)
    else:
        # Start with the fixed points only:
        n_initial, n_fixed = 0, fixed.shape[0]
        sum_dists_to_selected = cdist(props_scaled, fixed_scaled).sum(axis=1)

    # CORE of SUPPORT POINTS: select from points that minimize the energy-distance objective
    for n_selected in range(n_initial, n_pnts):
        obj = expected_sample_dist - sum_dists_to_selected / (n_selected + n_fixed)
        obj[i_selected] = inf
        # stochastic selection based on objective value...
        i_selected.append(_stochastic_pick(obj, stoch_frac, rng))
        # increment only the necessary contribution...
        sum_dists_to_selected += props_distances[:, i_selected[-1]].reshape(-1)

    if return_weights:
        # Calculate weights based on the number of nearest neighbors (approx. Voronoi cell vol.):
        #  <Referred to in the optimal-quantization literature as: Wasserstein-optimal weights.>
        i_nearest = props_distances[:, i_selected].argmin(axis=1)
        counts = bincount(i_nearest.astype('int64'), minlength=n_pnts)
        weights = (n_pnts / n_proposals) * counts

    # Setup the return arrays according to the input preferences:
    if return_by_index:
        ret_val = i_selected
    else:
        ret_val = proposals[i_selected]
    if return_weights:
        ret_val = (ret_val, weights)
    return ret_val


# Combine independent groups of sampled parameters & space-fill by sub-sampling:

def _tile_indices(size: int, n_out: int, rng: Generator) -> NDArray:
    """Create indices of length `n_out` by repeatedly sampling from [0, size) w/o replacement."""
    out = full(n_out, 0, dtype='int64')
    filled = 0
    while filled < n_out:
        perm = rng.permutation(size)
        take = min(size, n_out - filled)
        out[filled:filled + take] = perm[:take]
        filled += take
    return out


def _restructure_subsets(subsets: Subsets) -> tuple[list[NDArray], list, list[Optional[tuple]], bool]:
    """
    Accept `subsets` as a list or dict whose items are ndarrays or pandas DataFrames.
    Return in the form `subset_arrays`, `subset_keys`, `column_names`:
      subset_arrays - list of 2-D float ndarrays (the sample values),
      subset_keys   - list of labels (dict keys, else integer positions),
      column_names  - list of (columns, index_name) for items that were DataFrames (else None),
                      used to reassemble a DataFrame output; `is_frame` = any(meta).
    """
    if isinstance(subsets, dict):
        is_dict = True
        subset_keys = list(subsets.keys())
        items = list(subsets.values())
    else:
        is_dict = False
        items = list(subsets)
        subset_keys = list(range(len(items)))
    subset_arrays, column_names = [], []
    for it in items:
        if isinstance(it, pd.DataFrame):
            column_names.append((it.columns, it.index.name))
            subset_arrays.append(it.to_numpy())
        else:
            column_names.append(None)
            subset_arrays.append(array(it))
    return subset_arrays, subset_keys, column_names, is_dict


def _align_ln_pdfs(subset_ln_pdfs: LnPdfs, subset_keys: list, is_dict: bool) -> list[Optional[NDArray]]:
    """
    Align `subset_ln_pdfs` (None, list, or dict keyed like `subsets`) to the `subset_keys` order,
    returning a list of per-subset arrays (or None) with values coerced to ndarray.
    """
    if subset_ln_pdfs is None:
        return [None] * len(subset_keys)
    if isinstance(subset_ln_pdfs, dict):
        return [None if subset_ln_pdfs.get(key) is None else array(subset_ln_pdfs[key])
                for key in subset_keys]
    return [None if lp is None else array(lp) for lp in subset_ln_pdfs]


def _restructure_ouput(subset_arrays: list[NDArray], subset_keys: list,
                       column_names: list[Optional[tuple]], is_dict: bool,
                       subset_idx: list[NDArray],
                       i_fin: Optional[NDArray] = None) -> Union[NDArray, pd.DataFrame]:
    """
    Reassemble the combined design.  `subset_idx[i]` are indices into subset_arrays[i]; `i_fin`
    (JOINT mode) further sub-selects the aligned proposals, or None (already length-D).
    Returns a DataFrame (MultiIndex columns if keyed & any item was a DataFrame) when any
    input item was a DataFrame, otherwise a concatenated ndarray.
    """
    def _rows(i):
        idx = subset_idx[i] if i_fin is None else subset_idx[i][i_fin]
        return subset_arrays[i][idx]

    if any(fm is not None for fm in column_names):
        blocks = []
        for i, fm in enumerate(column_names):
            cols = fm[0] if fm is not None else range(subset_arrays[i].shape[1])
            blocks.append(pd.DataFrame(_rows(i), columns=cols))
        combined = pd.concat(blocks, axis=1, keys=subset_keys)  # MultiIndex: (label, column)
        return combined
    return concatenate([_rows(i) for i in range(len(subset_arrays))], axis=1)


def spacefill_permute_design(n_pnts: int, subsets: Subsets, subset_ln_pdfs: LnPdfs = None,
                             spacefill_method: SpacefillMethod = greedy_maximin_design,
                             spacefill_kwargs: Optional[dict] = None,
                             return_by_index: bool = False,
                             rng: Optional[Generator] = None
                             ) -> Union[NDArray, pd.DataFrame, list, dict]:
    """
    Combine independent `subsets` into a single `n_pnts`-point design by space-filling EACH
    submodel individually to `n_pnts`, then combining by permutation.  This gives zero
    duplication & optimal per-submodel marginal coverage — the right choice when joint
    interaction effects are small and the emulator exploits low-order (e.g. co-linear)
    structure.  (When joint space-filling matters, e.g. for GP emulation, use
    `combine_spacefill_design` instead.)

    `subsets` may be a list or a dict; its items may be ndarrays or pandas DataFrames (an
    MCMC-style sampling of one submodel's parameters, each).  With a dict, keys label the
    output; with DataFrame items, the output is a DataFrame (MultiIndex columns if keyed).
    `subset_ln_pdfs`, when given, aligns to `subsets` (same ordering / keys as the values).

    An under-supplied submodel (fewer than `n_pnts` samples) is tiled up to `n_pnts`, forcing
    some duplication; a warning reports the floor.  `spacefill_method` (w/ `spacefill_kwargs`,
    e.g. `scale`/`beta`/`stoch_frac`) performs each per-submodel selection; pass `subset_ln_pdfs`
    plus a `beta` in the kwargs to temper.  See `docs/spacefill.md`.
    """
    rng = default_rng() if rng is None else rng
    sf_kwargs = {} if spacefill_kwargs is None else dict(spacefill_kwargs)
    subset_arrays, subset_keys, column_names, is_dict = _restructure_subsets(subsets)
    k, D = len(subset_arrays), n_pnts
    ln_pdfs = _align_ln_pdfs(subset_ln_pdfs, subset_keys, is_dict)

    subset_idx = []
    for i, subset in enumerate(subset_arrays):
        m_i = subset.shape[0]
        if m_i >= D:
            idx = array(spacefill_method(D, subset, ln_pdfs=ln_pdfs[i], return_by_index=True,
                                         rng=rng, **sf_kwargs), dtype='int64')
            rng.shuffle(idx)  # unorder, so the permute isn't aligned
        else:
            warn(f"subset {subset_keys[i]!r} has only {m_i} samples for a design of {D} points: "
                 f"forcing at least {1 - m_i / D:.1%} duplication.")
            idx = _tile_indices(m_i, D, rng)  # under-supplied: tile up (reshuffled)
        subset_idx.append(idx)

    if return_by_index:
        return dict(zip(subset_keys, subset_idx)) if is_dict else subset_idx
    return _restructure_ouput(subset_arrays, subset_keys, column_names, is_dict, subset_idx)


def combine_spacefill_design(n_pnts: int, subsets: Subsets, subset_ln_pdfs: LnPdfs = None,
                             f_tol: float = 0.05, max_proposals: int = 100_000,
                             spacefill_method: SpacefillMethod = greedy_maximin_design,
                             spacefill_kwargs: Optional[dict] = None,
                             return_by_index: bool = False, verbose: bool = False,
                             rng: Optional[Generator] = None
                             ) -> Union[NDArray, pd.DataFrame, list, dict]:
    """
    Combine independent `subsets` into a single `n_pnts`-point design by tiling them into joint
    proposals, then space-filling the joint set — favoring joint space-filling quality (the right
    choice for e.g. GP emulation, where emulator uncertainty depends on joint distances).  (When
    only per-submodel marginal coverage matters, use `spacefill_permute_design` instead.)

    `subsets` may be a list or a dict; its items may be ndarrays or pandas DataFrames (an
    MCMC-style sampling of one submodel's parameters, each).  With a dict, keys label the output;
    with DataFrame items, the output is a DataFrame (MultiIndex columns if keyed).  `subset_ln_pdfs`,
    when given, aligns to `subsets` (same ordering / keys as the values).

    `f_tol` is the *approximate target* fraction of final points that repeat a submodel's sample
    (a balance knob trading joint-fill against duplication, NOT a hard constraint).  Realized
    duplication generally lands at or below `f_tol` — EXCEPT (i) when a submodel's supply is too
    small (fewer than `n_pnts` samples), forcing more by information content (see warning), or
    (ii) for strongly peaked marginals w/ `beta=0`, where maximin's tail-seeking inflates it
    (raise `beta` to mitigate).  The model predicts duplication only within ~2x across the variety
    of real distributions; for tighter control, iterate on the realized `f`.

    `max_proposals` caps the joint proposal count (the Mitchell/pairwise cost scales w/ it).
    `spacefill_method` (w/ `spacefill_kwargs`, e.g. `scale`/`beta`/`stoch_frac`) performs both the
    per-submodel down-select & the final selection; pass `subset_ln_pdfs` plus a `beta` in the
    kwargs to temper (independent submodel log-densities are summed for the joint).  `verbose`
    reports per-submodel m*, R, α, and estimated f.  See `docs/spacefill.md`.
    """
    rng = default_rng() if rng is None else rng
    sf_kwargs = {} if spacefill_kwargs is None else dict(spacefill_kwargs)
    subset_arrays, subset_keys, column_names, is_dict = _restructure_subsets(subsets)
    k, D = len(subset_arrays), n_pnts
    ln_pdfs = _align_ln_pdfs(subset_ln_pdfs, subset_keys, is_dict)
    m = [s.shape[0] for s in subset_arrays]

    for i, m_i in enumerate(m):
        if m_i < D:
            warn(f"subset {subset_keys[i]!r} has only {m_i} samples for a design of {D} points: "
                 f"marginal distinctness is capped at {m_i}, forcing at least {1 - m_i / D:.1%} "
                 f"duplication (regardless of `f_tol`={f_tol}).")

    def _estimate_f(R, α, γ=0.898):
        """Estimated per-submodel duplication: occupancy model x empirical maximin over-selection
        factor c(R) = 1 + γ(1 - 1/R)  (c -> 1 as R -> 1); see `docs/spacefill.md`."""
        if R <= 1:
            return 0.0
        base = 1 - 1 / (α * R)
        if base <= 0:     # α*R <= 1: fewer distinct candidates than selections -> saturated
            return 1 - α  #           (occupancy limit: at most α*D distinct of D -> f = 1 - α)
        return (1 + γ * (1 - 1 / R)) * (1 - α * (1 - base ** R))

    # Pick replication R for the binding submodel so estimated duplication ≈ f_tol (f monotone in
    # R, so bisect); fall back to the max_proposals cap when slack & to the forced floor 1/α_max:
    m_min, α_max = min(m), min(m) / D
    R_cap = max_proposals / m_min
    if α_max < 1.0:
        # Under-supplied binding subset: the forced-replication floor dominates; skip the budget
        # bisection (f_tol is unreachable — already warned) and just satisfy P >= D:
        R = 1.0 / α_max
    elif f_tol <= 0.0 or _estimate_f(R_cap, α_max) <= f_tol:
        R = R_cap
    else:
        lo, hi = 1.0, R_cap
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _estimate_f(mid, α_max) < f_tol else (lo, mid)
        R = 0.5 * (lo + hi)
    R = max(R, 1.0 / α_max)
    P = min(int(round(R * m_min)), max_proposals)

    # Per subset: random down-select to the shared pool (preserve candidate density), tile to P:
    subset_idx, report = [], []
    for i, subset in enumerate(subset_arrays):
        m_i = m[i]
        m_star = min(P, m_i)
        keep = rng.choice(m_i, m_star, replace=False) if m_star < m_i else arange(m_i)
        subset_idx.append(keep[_tile_indices(m_star, P, rng)])
        report.append((m_star, P / m_star, m_star / D, _estimate_f(P / m_star, m_star / D)))

    proposals = concatenate([subset_arrays[i][subset_idx[i]] for i in range(k)], axis=1)
    ln_pdf = (sum(ln_pdfs[i][subset_idx[i]] for i in range(k))
              if all(lp is not None for lp in ln_pdfs) else None)

    if verbose:
        print(f"[combine_spacefill_design] method = {spacefill_method.__name__}")
        print(f"  D = {D}, m = {m}, m_min = {m_min}, α_max = {α_max:.3g}, f_tol = {f_tol}, P = {P}")
        print(f"  {'subset':>8} {'m':>8} {'m*':>8} {'R':>7} {'α':>7} {'f_est':>9}")
        for i, (m_star, R_i, α_i, f_i) in enumerate(report):
            print(f"  {str(subset_keys[i]):>8} {m[i]:>8} {m_star:>8} {R_i:>7.3g} {α_i:>7.3g} "
                  f"{f_i:>9.3%}")

    i_fin = array(spacefill_method(D, proposals, ln_pdfs=ln_pdf, return_by_index=True, rng=rng,
                                   **sf_kwargs), dtype='int64')
    if return_by_index:
        sel = [subset_idx[i][i_fin] for i in range(k)]
        return dict(zip(subset_keys, sel)) if is_dict else sel
    return _restructure_ouput(subset_arrays, subset_keys, column_names, is_dict, subset_idx, i_fin)
