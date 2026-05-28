"""
Sean's note: This is an iteration on (and a simplification of) utilities in
             `~/Documents-Utah/Python/design_of_experiments` — namely `doe_utilities.py` &
             `DOE_spacefilling.py`.  Notably, those versions allowed pre-existing (fixed) points.
             That feature can be re-added if we find value in the active-learning techniques.
"""
from math import ceil, log2
from numpy import array, full, linspace, arange, tril_indices, unravel_index, minimum, unique, exp, inf
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from scipy.stats import qmc

# Generative space-filling designs on a unit box:

def lhs_design(n_pnts, n_dims, position="center", rng=None):
    """
    Sample a latin hypercube design (a space-filling design which ensures a coordinate-aligned
    stratified uniformity) over the unit box w/ `n_pnts` number of points & `n_dims` number of
    dimensions.  The position of each point within it sub-region can be specified as
    'center' (default), 'edges', or 'random'.  Developed by Mike Mckay at Los Alamos in 1979.
    """
    if rng is None:
        rng = default_rng()
    rand = rng.random
    permute = rng.permutation
    δx = 1 / n_pnts
    endp = (position == "edges")
    design = array([permute(linspace(0, 1, n_pnts, endpoint=endp)) for dim in range(n_dims)]).T
    if position == "center":
        design += δx / 2
    elif position == "random":
        design += δx * rand((n_pnts, n_dims))
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
    return sp_lhs(n=n_pnts)


def maximin_of_designs(n_pnts, n_dims, n_samples, sampler=lhs_design, **kwargs):
    """
    Optimize a space-filling property of designs by sampling `n_samples` number of designs from
    `sampler(n_pnts, n_dims, **kwargs)` over the unit box with `n_pnts` number of points and
    `n_dims` number of dimensions — and select the one that maximizes the minimum distance within
    the design (search for designs w/ larger closest-pair spacing).
    """
    def min_dist_count(design):
        """
        For a specified `design`, calculate the minimum inter-point distance & the number of times
        that distance is repeated.
        """
        dist_array = cdist(design, design)
        dist_vals = dist_array[*tril_indices(design.shape[0], -1)]
        min_dist = dist_vals.min()
        min_count = (dist_vals == min_dist).sum()
        return min_dist, min_count

    design = None
    maximin_dist = -inf
    maximin_count = 0
    for i in range(n_samples):
        proposed = sampler(n_pnts, n_dims, **kwargs)
        min_dist, min_count = min_dist_count(proposed)
        if min_dist > maximin_dist  or  min_dist == maximin_dist and min_count < maximin_count:
            design, maximin_dist, maximin_count = proposed, min_dist, maximin_count
    return design


def sobol_design(n_pnts, n_dims, rng=None, **kwargs):
    """
    Sample a Sobol-sequence design (a space-filling desigh which aims for low discrepancy) over
    the unit box with `n_pnts` number of points & `n_dims` number of dimensions — `n_pnts` must
    be an integer power of 2.
    (This is simply a wrapper on `scipy.stats.qmc.Sobol` for the sake of a uniform call signature.)
    """
    sp_sobol = qmc.Sobol(d=n_dims, rng=rng, **kwargs)
    return sp_sobol.random_base2(ceil(log2(n_pnts)))


# Sub-sampling designs for arbitrary densities:

def greedy_maximin_design(n_pnts, n_dims, n_proposal=None, proposal=None, return_weights=False,
                          rng=None):
    """
    Sample a space-filling design w/ `n_pnts` number of points & `n_dims` number of dimensions that
    is ordered by greedily sub-selecting points from a larger `proposal` set (which if not provided,
    will be uniformly random over the unit box w/ `n_proposal` number of points) according to the
    best maximin value.
    Variations go by different names, but all use the same principles & arrive at similar results:
    - The baseline additive approach is referred to by statisticians as "greedy maximin sampling",
    - The baseline subtractive approach is referred to by computer scientists as "progressive
      Poisson-disk sampling" or "blue-noise sampling",
    Although there is a large variety of valid algorithms, one caution is taken here to avoid a
    naive implementation resulting in computation of O(n_pnts^2 * n_proposal * n_dims) in favor of
    one that gives O(n_pnts * n_proposal * n_dims).  (Mitchell 1991 demonstrated the first
    O(n_pnts * n_proposal * n_dims) algorithm, and Bridson 2007 went further by demonstrating an
    O(n_pnts * n_dims) algorithm — but that is overkill for the current intent.  So, this
    implementation opts for more interpretability.)
    """
    if rng is None:
        rng = default_rng()
    rand = rng.random
    choice = rng.choice
    if proposal is None:
        proposal = prop = rand((n_proposal, n_dims))
    else:
        p_max, p_min = proposal.max(axis=0, keepdims=True), proposal.min(axis=0, keepdims=True)
        prop = (proposal - p_min) / (p_max - p_min + 1e-12)  # min/max scaled version of `proposal`
        n_proposal = proposal.shape[0]

    # Initialize by selecting the two most distant points in a small subset of `proposal`:
    i_init = choice(arange(n_proposal), size=max(n_pnts // 3, min(n_proposal, 20)), replace=False)
    dist_init = cdist(prop[i_init], prop[i_init])
    i_design = [int(arg) for arg in unravel_index(dist_init.argmax(), dist_init.shape)]
    # For each point in `proposal`, find the index of & distance to the nearest of those selected:
    i_nearest = full(n_proposal, n_proposal, dtype='uint')
    dist2nearest = full(n_proposal, inf, dtype='f8')
    for i_selected in i_design:
        # minimum(dist2nearest, cdist(prop[[i_selected]], prop).reshape(-1), out=dist2nearest)
        dist2selected = cdist(prop[[i_selected]], prop).reshape(-1)
        update_mask = dist2selected < dist2nearest
        i_nearest[update_mask] = i_selected
        dist2nearest[update_mask] = dist2selected[update_mask]
        # ensure each selected point is its own nearest selected point...
        i_nearest[i_selected] = i_selected
        # prevent selected points from being selected again...
        dist2nearest[i_selected] = -inf
    # Greedily iterate through the remaining `n_pnts` number of design points:
    for i in range(2, n_pnts):
        # identify the index of the next maximin point...
        i_maximin = int(dist2nearest.argmax())
        i_design.append(i_maximin)
        # update `i_nearest` & `dist2nearest` by comparing only to the newly selected point...
        # minimum(dist2nearest, cdist(prop[[i_maximin]], prop).reshape(-1), out=dist2nearest)
        dist2maximin = cdist(prop[[i_maximin]], prop).reshape(-1)
        update_mask = dist2maximin < dist2nearest
        i_nearest[update_mask] = i_maximin
        dist2nearest[update_mask] = dist2maximin[update_mask]
        i_nearest[i_maximin] = i_maximin  # (selected point is its own nearest selected)
        dist2nearest[i_maximin] = -inf  # (prevent selected points from being selected again)
    if not return_weights:
        return proposal[i_design]
    else:
        ind, counts = unique(i_nearest, return_counts=True)
        all_counts = full(n_proposal, 0, dtype='uint')
        all_counts[ind] = counts
        weights = (n_pnts / n_proposal) * all_counts
        return proposal[i_design], weights[i_design]


def support_points_design(n_pnts, samples, weights=None,
                          scale="std", n_props=None, stoch_frac=0.1, rng=None):
    """
    Sub-select a space-filling support-points design w/ `n_pnts` number of points from the
    representative Monte-Carlo `samples` with optionally associated importance `weights`.
    This is a stochastic greedy optimization algorithm applied to the support-points energy-
    distance objective.  Scaling of the provided `samples` may be `None`, "std" or "range".
    For the sake of computation only a subset of `samples` are proposed for consideration, the
    number of which can be specified as `n_props`.  The degree of stochasticity in the greedy
    selection can be adjusted by specifying the fraction of top samples to consider as `stoch_frac`.
    """
    if rng is None:
        rng = default_rng()
    choice = rng.choice
    n_samps, n_dims = samples.shape

    if weights is None:
        weights = full(n_samps, 1, dtype='f8')

    if scale is None:
        samples_scaled = samples
    elif scale == "std":
        s_mean, s_std = samples.mean(axis=0, keepdims=True), samples.std(axis=0, keepdims=True)
        samples_scaled = (samples - s_mean) / (s_std + 1e-12)
    elif scale == "range":
        s_min, s_max = samples.min(axis=0, keepdims=True), samples.max(axis=0, keepdims=True)
        samples_scaled = (samples - s_min) / (s_max - s_min + 1e-12)

    # Select the proposed subset of `samples`:
    if n_props is None:
        n_props = min(10 * n_pnts, n_samps)
    proposal_indices = choice(n_samps, size=n_props, replace=False)
    proposals = samples_scaled[proposal_indices]

    # Precompute each sample's expected distance (reused on each iteration):
    expected_sample_dist = (weights @ cdist(samples_scaled, proposals)) / weights.sum()

    # For the first point select one w/ an expected distance in the lower 50 percentile:
    selected_indices = []
    n_stoch = max(int(stoch_frac * n_props), 1)
    selected_indices.append(choice(expected_sample_dist.argpartition(n_props // 2)[:n_props // 2]))
    sum_of_dists_to_selected = cdist(proposals, proposals[selected_indices]).reshape(-1)

    # Greedily selected each subsequent remaining point w/ limited stochasticity:
    for n in range(1, n_pnts):
        n_stoch = max(int(stoch_frac * (n_props - n)), 1)
        # Calculate the support-points energy-distance objective:
        obj = expected_sample_dist - sum_of_dists_to_selected / n
        obj[selected_indices] = inf
        # Select from the lowest `n_stoch` values w/ a Boltzmann/softmax weighting:
        i_best = obj.argpartition(n_stoch)[:n_stoch]
        p_best = exp(-(obj[i_best] - obj[i_best].min()) / (obj[i_best].std() + 1e-12))
        p_best /= p_best.sum() 
        selected_indices.append(choice(i_best, p=p_best))
        # Update `sum_of_dists_to_selected`:
        sum_of_dists_to_selected += cdist(proposals, proposals[[selected_indices[-1]]]).reshape(-1)
        
    return samples[proposal_indices[selected_indices]]