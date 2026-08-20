"""
This is a simple toolkit to perform design-of-experiment sampling according to my preferred
methods - Latin hyper-cube sampling, Sobol sampling & greedy maximin subsampling.

Created April 2017 @author: Sean T. Smith, updated July-Aug 2026.
"""

from math import ceil, log2
from numpy import (array, full, linspace, arange, concatenate, tril_indices, unravel_index,
                   bincount, isfinite, outer, sqrt, exp, log, inf)
from numpy.linalg import svd
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from scipy.stats import qmc


# Generative space-filling designs on a unit box:

def lhs_design(n_pnts, n_dims, fixed=None, position="center", rng=None):
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
        available_strata = list(linspace(0, 1, n_tot, endpoint=include_endpoint))
        if fixed is not None:
            offset = 0 if include_endpoint else δx / 2
            for coordinate in fixed[:, dim]:
                occupied_position = coordinate - offset
                i_occupied = min(range(len(available_strata)),
                                 key=lambda i: abs(available_strata[i] - occupied_position))
                available_strata.pop(i_occupied)
        all_strata.append(rng.permutation(available_strata))
    design = array(all_strata).T
    if position == "center":
        design += δx / 2
    elif position == "random":
        design += δx * rng.random((n_pnts, n_dims))
    return design


def oa_lhs_design(n_pnts, n_dims, rng=None):
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


def sobol_design(n_pnts, n_dims, rng=None, **kwargs):
    """
    Sample a Sobol-sequence design (a space-filling design which aims for low discrepancy) over
    the unit box with `n_pnts` number of points & `n_dims` number of dimensions — `n_pnts` must
    be an integer power of 2.
    (This is simply a wrapper on `scipy.stats.qmc.Sobol` for the sake of a uniform call signature.)
    """
    sp_sobol = qmc.Sobol(d=n_dims, rng=rng, **kwargs)
    return sp_sobol.random_base2(ceil(log2(n_pnts)))


def optimal_of_designs(n_pnts, n_dims, n_samples, sampler=lhs_design, optimality="maximin", 
                       test_pnts=None, pnts2bases=None, **kwargs):
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

def _scale_util(proposals, fixed, scale):
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

def _stochastic_pick(objective, stoch_frac, rng):
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


def greedy_maximin_design(n_pnts, proposals, ln_pdfs=None, fixed=None, scale="cov", beta=0.0,
                          stoch_frac=0.1, return_weights=False, return_by_index=False, rng=None):
    """
    Create a space-filling design w/ `n_pnts` number of output points that is selected in order by
    greedily sub-selecting points from a larger `proposals` set according to best maximin value.
    If the log values of the unscaled distribution are provided for each point, the algorithm is
    modified to density-weighted & tempered greedy maximin (with tempering parameter `beta` — where
    `beta=0.0` reduces to the unweighted algorithm, and `beta=1.0` will result in samples that are
    approximately representative of the distribution).  Existing `fixed` points can be provided.
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

    # Density weighting: higher-density & larger beta -> smaller length-scale -> further spaced
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
        counts = bincount(i_nearest, minlength=n_proposals)[i_selected]
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


def support_points_design(n_pnts, proposals, ln_pdfs=None, fixed=None, scale="cov", beta=1.0,
                          stoch_frac=0.1, return_weights=False, return_by_index=False, rng=None):
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
        counts = bincount(i_nearest, minlength=n_pnts)
        weights = (n_pnts / n_proposals) * counts

    # Setup the return arrays according to the input preferences:
    if return_by_index:
        ret_val = i_selected
    else:
        ret_val = proposals[i_selected]
    if return_weights:
        ret_val = (ret_val, weights)
    return ret_val