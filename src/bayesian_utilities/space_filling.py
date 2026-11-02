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
    Sample a Latin-hypercube design over the unit box — a space-filling design with uniform,
    coordinate-aligned stratification (one point per stratum in every dimension).

    Parameters
    ----------
    n_pnts, n_dims : int
        Number of points and dimensions of the design.
    fixed : ndarray, optional
        Existing points whose occupied strata are excluded, so the new points complete (or
        approximately augment) a `len(fixed) + n_pnts` Latin hypercube.
    position : {"center", "edges", "random"}
        Placement of each point within its stratum.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray of shape (n_pnts, n_dims)

    See Also
    --------
    docs/space_filling.md : background and method-selection guidance.
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
    Sample a strength-2 orthogonal-array Latin-hypercube design over the unit box — a space-filling
    design with pairwise stratified uniformity.  `n_pnts` must be the square of a prime that is
    at least `n_dims - 1`.  (A thin wrapper on `scipy.stats.qmc.LatinHypercube` for a uniform API;
    for strength > 2, Sobol is generally more performant.)

    Returns
    -------
    ndarray of shape (n_pnts, n_dims)
    """
    sp_lhs = qmc.LatinHypercube(d=n_dims, strength=2, rng=rng)
    return sp_lhs.random(n=n_pnts)


def sobol_design(n_pnts: int, n_dims: int, rng: Optional[Generator] = None, **kwargs) -> NDArray:
    """
    Sample a Sobol-sequence design over the unit box — a low-discrepancy space-filling design.
    `n_pnts` must be an integer power of 2.  (A thin wrapper on `scipy.stats.qmc.Sobol` for a
    uniform API; extra keyword arguments are forwarded.)

    Returns
    -------
    ndarray of shape (n_pnts, n_dims)
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
    Draw `n_samples` candidate designs from `sampler(n_pnts, n_dims, **kwargs)` and return the one
    best satisfying the chosen `optimality` criterion.

    Parameters
    ----------
    n_pnts, n_dims : int
        Size and dimensionality of each candidate design.
    n_samples : int
        Number of candidate designs to draw.
    sampler : callable
        Generative design function, called as `sampler(n_pnts, n_dims, **kwargs)`.
    optimality : {"maximin", "D", "A", "I", "G"}
        - "maximin" : maximize the smallest inter-point distance (ties broken toward fewer
          repeats of that distance).
        - "D" : maximize differential posterior Shannon information, max det(XᵀX).
        - "A" : minimize average posterior variance of the linear parameters, min trace((XᵀX)⁻¹).
        - "I" : minimize the average prediction uncertainty across `test_pnts`.
        - "G" : minimize the largest prediction uncertainty among `test_pnts`.
    test_pnts : ndarray, optional
        Evaluation points for the "I" and "G" criteria (defaults to the design itself).
    pnts2bases : callable, optional
        Maps points to a regression basis matrix X; required for "D", "A", "I", "G".

    Returns
    -------
    ndarray of shape (n_pnts, n_dims)
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
    """Scale `proposals` (and `fixed`, if given) by: None, "range", "std", or "cov" (whitening)."""
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
    Pick an index minimizing `objective`, softmax-sampled over its best `stoch_frac` fraction
    (temperature scales with `stoch_frac`); deterministic argmin when `stoch_frac == 0`.
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
    Sub-select a space-filling design that greedily identifies `n_pnts` points from a larger
    `proposals` set by repeatedly taking the point farthest from those already chosen
    (greedy maximin, a.k.a. progressive Poisson-disk / blue-noise sampling).

    Supplying `ln_pdfs` activates density-weighted, tempered selection: `beta=0` is the unweighted
    maximin, `beta=1` yields a sample approximately representative of the distribution.

    Parameters
    ----------
    n_pnts : int
        Number of points to select.
    proposals : ndarray of shape (n_proposals, n_dims)
        Candidate points to sub-select from.
    ln_pdfs : ndarray of shape (n_proposals,), optional
        Log of the (unnormalized) density at each proposal; enables tempering via `beta`.
    fixed : ndarray, optional
        Pre-existing points that the selection must space-fill around.
    fixed_ln_pdfs : ndarray, optional
        Log-densities for `fixed`; required when both `ln_pdfs` and `fixed` are given, and must
        use the same distribution and normalization as `ln_pdfs`.
    scale : {None, "range", "std", "cov"}
        Pre-scaling of the space in which distances are measured.
    beta : float
        Tempering parameter (see above); only active when `ln_pdfs` is given.
    stoch_frac : float
        Stochasticity of the greedy step: 0 is deterministic; up to 1 widens the candidate pool.
    return_weights : bool
        Also return approximate Voronoi-cell weights (sum to `n_pnts`); when `fixed` is given,
        returned as a `(fixed_weights, selected_weights)` tuple.
    return_by_index : bool
        Return indices into `proposals` rather than the point coordinates.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray or list of int
        The selected points (or their indices if `return_by_index`), optionally paired with
        weights if `return_weights`.

    See Also
    --------
    docs/space_filling.md : factors affecting behavior, tempering guidance, and the algorithmic
        complexity considerations (Mitchell 1991 / Bridson 2007).
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
    Sub-select a space-filling design that greedily identifies `n_pnts` points from a larger
    `proposals` set to minimize an energy-distance (support-points) objective, yielding a
    selection representative of the proposal distribution.

    Supplying `ln_pdfs` activates tempering: `beta=1` is the untempered objective; `beta->0`
    flattens the target distribution, favoring the most distant (tail) points.

    Parameters
    ----------
    n_pnts : int
        Number of points to select.
    proposals : ndarray of shape (n_proposals, n_dims)
        Candidate points to sub-select from.
    ln_pdfs : ndarray of shape (n_proposals,), optional
        Log of the (unnormalized) density at each proposal; enables tempering via `beta`.
    fixed : ndarray, optional
        Pre-existing points to select around.  Note: `fixed` is not assumed representative of the
        distribution, so unless `n_pnts >> len(fixed)` the result may also not be representative.
    scale : {None, "range", "std", "cov"}
        Pre-scaling of the space in which distances are measured.
    beta : float
        Tempering parameter (see above); only active when `ln_pdfs` is given.
    stoch_frac : float
        Stochasticity of the greedy step: 0 is deterministic; up to 1 widens the candidate pool.
    return_weights : bool
        Also return approximate Voronoi-cell weights (sum to `n_pnts`).
    return_by_index : bool
        Return indices into `proposals` rather than the point coordinates.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray or list of int
        The selected points (or their indices if `return_by_index`), optionally paired with
        weights if `return_weights`.

    See Also
    --------
    docs/space_filling.md : the ~40k-proposal performance ceiling, tempering guidance, and
        regime-dependent recommendations relative to greedy maximin.
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
    """Draw indices of length `n_out`, by repeatedly permuting [0, size) (sampling w/o repl.)."""
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
    Restructure list/dict `subsets` of ndarray/DataFrame items into
    (arrays, keys, column_meta, is_dict) for uniform downstream handling.
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
    """Align `subset_ln_pdfs` (None/list/dict) to `subset_keys` order as a list of arrays/None."""
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
    Reassemble the combined design from per-subset indices; returns a DataFrame (MultiIndex
    columns if keyed) when any input item was a DataFrame, else a concatenated ndarray.
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
    Combine independent `subsets` (separately-sampled parameter blocks) into one `n_pnts`-point
    design by space-filling each block individually to `n_pnts`, then combining by permutation.

    Gives zero within-block duplication and optimal per-block marginal coverage — preferred when
    joint interaction effects are weak (e.g. an emulator exploiting low-order / co-linear
    structure).  Use `combine_spacefill_design` instead when joint space-filling matters (e.g. GP
    emulation).

    Parameters
    ----------
    n_pnts : int
        Size of the combined design.
    subsets : list or dict of (ndarray or DataFrame)
        The independent sample blocks.  A dict labels the output by its keys; DataFrame items make
        the output a DataFrame (with MultiIndex columns if keyed).
    subset_ln_pdfs : list or dict, optional
        Per-block log-densities aligned to `subsets`; enables tempering (with a `beta` in
        `spacefill_kwargs`).
    spacefill_method : callable
        Per-block sub-sampling design (`greedy_maximin_design` or `support_points_design`).
    spacefill_kwargs : dict, optional
        Options forwarded to `spacefill_method` (e.g. `scale`, `beta`, `stoch_frac`).
    return_by_index : bool
        Return per-block indices into `subsets` (mirroring list/dict input) instead of values.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray, DataFrame, list, or dict
        The combined design, or per-block selection indices if `return_by_index`.

    Warns
    -----
    An under-supplied block (fewer than `n_pnts` samples) is tiled up, forcing duplication; the
    floor `1 - m/n_pnts` is reported.

    See Also
    --------
    docs/space_filling.md : the combine-vs-permute trade-off and which to use.
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
    Combine independent `subsets` (separately-sampled parameter blocks) into one `n_pnts`-point
    design by tiling the blocks into joint proposals, then space-filling the joint set.

    Favors joint space-filling quality — preferred when joint inter-point distances matter (e.g.
    GP emulation).  Use `spacefill_permute_design` instead when only per-block marginal coverage
    is needed.

    Parameters
    ----------
    n_pnts : int
        Size of the combined design.
    subsets : list or dict of (ndarray or DataFrame)
        The independent sample blocks.  A dict labels the output by its keys; DataFrame items make
        the output a DataFrame (with MultiIndex columns if keyed).
    subset_ln_pdfs : list or dict, optional
        Per-block log-densities aligned to `subsets`; summed into a joint log-density to enable
        tempering (with a `beta` in `spacefill_kwargs`).
    f_tol : float
        Approximate target fraction of final points that repeat a block's sample (a balance knob
        trading joint-fill against duplication, NOT a hard constraint).  See Notes.
    max_proposals : int
        Cap on the joint proposal count (the pairwise selection cost scales with it).
    spacefill_method : callable
        Sub-sampling design used for both the per-block down-select and the final joint selection.
    spacefill_kwargs : dict, optional
        Options forwarded to `spacefill_method` (e.g. `scale`, `beta`, `stoch_frac`).
    return_by_index : bool
        Return per-block indices into `subsets` (mirroring list/dict input) instead of values.
    verbose : bool
        Print the per-block m*, replication R, abundance α, and estimated duplication f.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray, DataFrame, list, or dict
        The combined design, or per-block selection indices if `return_by_index`.

    Notes
    -----
    Realized duplication usually lands at or below `f_tol`, with two exceptions: (i) an
    under-supplied block (fewer than `n_pnts` samples) forces duplication by information content
    (a warning reports the floor `1 - m/n_pnts`); (ii) strongly peaked marginals selected with
    `beta=0` over-duplicate (raise `beta` to mitigate).  The duplication model is accurate only
    to within ~2x across varied distributions; for tight control, iterate on the realized f.

    See Also
    --------
    docs/space_filling.md : the duplication model, the combine-vs-permute trade-off, and the
        three supply regimes (abundant / deficient / intermediate).
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
