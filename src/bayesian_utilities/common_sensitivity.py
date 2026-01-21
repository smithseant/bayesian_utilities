"""
A Collection of routines for sensitivity analysis.
@author: Sean T. Smith, Jan-Feb 2023, Los Alamos National Laboratory
"""
from numpy import empty, zeros, arange, tile, expand_dims, unravel_index, maximum, abs, sign, sqrt
from numpy.linalg import eigh
from numpy.random import default_rng
my_rng = default_rng()
i_perm = my_rng.permutation
from scipy.stats.qmc import Sobol as SobolSamp

def path_dist(k_dims, path_a, path_b):
    """
    Calculate the quality metric between two paths as suggested in Saltelli 2007 (which in turn
    referenced Campolongo et. al. 2007).
    """
    dist = 0
    for i in range(k_dims + 1):
        for j in range(k_dims + 1):
            dist += sqrt(((path_a[i] - path_b[j])**2).sum())
    return dist

def morris(func, k_dims, p_disc, n_paths, *args, campolongo=False, verbose=False, **kwargs):
    """
    Perform the Morris one-at-a-time (a.k.a. elementary-effects) algorithm, as outlined in
    M. Morris. "Factor Sampling Plans for Preliminary Computational Experiments,"
    Technometrics, May 1991, vol. 33, no. 2.
    Parameters:
      func (callable),  function who's sensitivity will be interrogated (vectorized & appropriately
                        transformed & scaled so its parameters are uniform over the unit box),
      k_dims (int),  number of parameter dimensions,
      p_disc (int),  number of discretization points in each dimension,
      n_paths (int),  number of walked paths (each entailing k_dims + 1 function evaluations) in
                      the complete design,
      campolongo ([False] or int > n_paths, optional),  whether to perform a greedy optimization
                                                        for the sampled paths.
    Returns:
      μ_di (array[k_dims]),  sample mean of the elementary effects in each dimension,
      μ_adi (array[k_dims]), sample mean of the absolute elementary effects in each dimension,
      σ_di (array[k_dims]),  sample standard deviation of the el. eff. in each dim.
    """

    # Set up the design:
    δ = 1 / (p_disc - 1)  # discretization includes points at zero & one
    Δ = p_disc / (2 * (p_disc - 1))  # Morris's heuristic
    i_rand = lambda p, k: my_rng.integers(low=0, high=p, size=k)

    n_test = campolongo if campolongo else n_paths
    x_star = empty((n_test, k_dims))
    order = empty((n_test, k_dims), dtype='uint')
    order_lookup = empty((n_test, k_dims), dtype='uint')
    steps = zeros((n_test, (k_dims + 1), k_dims))
    path = empty((n_test, (k_dims + 1), k_dims))

    for n in range(n_test):
        x0 = δ * i_rand(p_disc, k_dims)
        step_sizes = -sign(x0 - 0.5) * Δ  # shortcut works when Δ > 0.5
        order[n] = i_perm(k_dims)  # sequence of jumps: index is order, value is dimension
        for i in range(k_dims):
            order_lookup[n, int(order[n, i])] = i  # index is dimension, value is order
        steps[n, arange(k_dims) + 1, order[n]] = step_sizes[order[n]]
        path[n] = x0 + steps[n].cumsum(axis=0)
        x_star[n] = x0 + steps[n, -1]

    if campolongo:
        # Calculate and store quality metrics
        dist = zeros((n_test, n_test))
        for i in range(n_test):
            for j in range(i + 1, n_test):
                dist[i, j] = dist[j, i] = path_dist(k_dims, path[i], path[j])
        # Greedy minimax optimization algorithm
        i_select = []  # indices of selected paths
        i_not = [i for i in range(n_test)]  # currently unselected indices
        i, j = unravel_index(dist.argmax(), dist.shape)
        i_not.remove(i)
        i_not.remove(j)
        i_select.extend([i, j])
        # Include the path with the next highest quality metric in relation to those selected
        for n in range(2, n_paths):
            val = dist[i_select][:, i_not].min(axis=0).max()
            for i in i_not:
                if val == dist[i_select].min(axis=0)[i]:
                    i_next = i
                    break
            i_not.remove(i_next)
            i_select.append(i_next)
        # Down select
        x_star = x_star[i_select]
        order = order[i_select]
        order_lookup = order_lookup[i_select]
        steps = steps[i_select]
        path = path[i_select]
        if verbose:
            print('Completed campolongo optimization')

    # Evaluate the function:
    n_feval = (k_dims + 1) * n_paths  # number of function evaluations
    f = func(path.reshape((n_feval, k_dims)), *args, **kwargs).reshape(path.shape[:2])

    # Perform the sensitivity analysis:
    d_ni = empty((n_paths, k_dims))  # each of the elementary effects
    for n in range(n_paths):
        d_ni[n] = ((f[n, order_lookup[n] + 1] - f[n, order_lookup[n]])
                   / steps[n, order_lookup[n] + 1, arange(k_dims)])
    μ_di = d_ni.mean(axis=0)
    μ_adi = abs(d_ni).mean(axis=0)
    σ_di = d_ni.std(axis=0)

    return μ_di, μ_adi, σ_di


def multi_L(func, k_dims, n_Ls, *args, **kwargs):
    """
    Perform Sean's modification to Morris' algorithm — this approach is a simplification in two
    ways. First, rather than introducing an intermediate discrete problem, just be willing to have
    limited coverage of the correctly sampled continuous problem. Second, rather than constructing
    a path that appears as an L when projected to any two dimensions and requires storing multiple
    variables to analyze, just use all the edges that connect to a single corner.
    Parameters:
      func (callable),  function who's sensitivity will be interrogated (vectorized & appropriately
                        transformed & scaled so its parameters are uniform over the unit box),
      k_dims (int),  number of parameter dimensions,
      n_Ls (int),  number of hyper-L designs (each entailing k_dims + 1 function evaluations).
    Returns:
      μ_di (array[k_dims]),  sample mean of the elementary effects in each dimension,
      σ_di (array[k_dims]),  sample standard deviation of the el. eff. in each dim.

    'A long habit of not thinking a thing wrong, gives it a superficial appearance of being right, and raises at first a formidable outcry in defense of custom. But the tumult soon subsides. Time makes more converts than reason.' -Thomas Paine
    """

    # Set up the design:
    x_star = my_rng.random((n_Ls, k_dims))
    x_opp = my_rng.random((n_Ls, k_dims))
    # TODO: Add options for LHD, Sobol, etc.
    Δ = x_opp - x_star
    Ls = tile(x_star.reshape((n_Ls, 1, k_dims)), (1, (k_dims + 1), 1))
    for j  in range(k_dims):
        Ls[:, (j + 1), j] = x_opp[:, j]

    # Evaluate the function:
    n_feval = (k_dims + 1) * n_Ls  # number of function evaluations
    f = func(Ls.reshape((-1, k_dims)), *args, **args).reshape(Ls.shape[:2])

    # Perform the sensitivity analysis:
    d_ni = (f[:, 1:] - f[:, 0].reshape((-1, 1))) / Δ  # each of the elementary effects
    μ_di = d_ni.mean(axis=0)
    μ_adi = abs(d_ni).mean(axis=0)
    σ_di = d_ni.std(axis=0)
    # TODO: perform SVD on d_ni — the singular values are the sqrt of Λ from V @ Λ @ V**-1 = d_ni.T @ d_ni
    # Λ, V = eigh(d_ni.T @ d_ni)

    return μ_di, μ_adi, σ_di

def sobol(func, k_dims, log2_nL, *args, **kwargs):
    """
    Perform Sobol's method for variance-based global sensitivity as outlined in his paper
    I.M. Sobol, "Global sensitivity indices for nonlinear mathematical models and their Monte
    Carlo estimates," Math. & Comp. in Sim., vol. 55, 2001, pp. 271-280.
    Only the univariate and bi-variate sensitivity indices ,with their 'total' counterparts,
    are implemented (no higher-order indices nor arbitrary subsets).
    """

    # Set up the design:
    n_Ls = 2**log2_nL
    SobolSampler = SobolSamp(d=(2 * k_dims), scramble=True) #, optimization='lloyd')
    AB = SobolSampler.random_base2(log2_nL)
    A = AB[:, :k_dims]
    B = AB[:, k_dims:]

    # Univariate (L) designs
    Ab1 = tile(expand_dims(A, 1), (1, k_dims, 1))
    for j in range(k_dims):
        Ab1[:, j, j] = B[:, j]

    # # Bi-variate (two step) designs
    # Ab2 = tile(expand_dims(A, 1), (1, int(k_dims * (k_dims - 1) / 2), 1))
    # j_Ab2 = 0
    # for i_B in range(k_dims):
    #     for j_B in range((i_B + 1), k_dims):
    #         Ab2[:, j_Ab2, i_B] = B[:, i_B]
    #         Ab2[:, j_Ab2, j_B] = B[:, j_B]
    #         j_Ab2 += 1
    # # This begins to look like full factorial designs (just smaller boxes).

    # Evaluate the function:
    n_feval = n_Ls * (2 + k_dims + k_dims * (k_dims - 1) / 2)
    f_A  = func(A, *args, **kwargs).reshape((-1, 1))
    f_B  = func(B, *args, **kwargs).reshape((-1, 1))
    f_Ab1 = func(Ab1.reshape((-1, k_dims)), *args, **kwargs).reshape((n_Ls, -1))
    # f_Ab2 = func(Ab2.reshape((-1, k_dims)), *args, **kwargs).reshape((n_Ls, -1))

    # Calculate the sensitivity indices:
    f0 = (f_A.sum() + f_B.sum()) / (2 * n_Ls)
    V = ((f_A**2).sum() + (f_B**2).sum()) / (2 * n_Ls) - f0**2

    # # Sobol 1991
    # V1 = (f_B * f_Ab1).mean(axis=0) - f0**2
    # V1tot = ((f_A - f_Ab1)**2).sum(axis=0) / (2 * n_Ls)
    # V2 = (f_B * f_Ab2).mean(axis=0) - f0**2

    # # Saltelli 2002 (as provided in Saltelli's 2007 book)
    # V1 = (f_B * f_Ab1).mean(axis=0) - f0**2
    # V1tot = (f_A * f_Ab1).mean(axis=0) - f0**2
    # V2 = (f_B * f_Ab2).mean(axis=0) - f0**2

    # Saltelli 2010 (as provided on wikipedia.org)
    V1 = ((f_Ab1 - f_A) * f_B).mean(axis=0)
    V1tot = ((f_Ab1 - f_A)**2).sum(axis=0) / (2 * n_Ls)
    # V2 = ((f_Ab2 - f_A) * f_B).mean(axis=0)

    V1 = maximum(V1, 0)
    S1 = V1 / V
    S1tot = V1tot / V
    # S2 = zeros((k_dims, k_dims))
    # k = 0
    # for i in range(k_dims):
    #     for j in range((i + 1), k_dims):
    #         S2[i, j] = S2[j, i] = maximum(V2[k] / V - (S1[i] + S1[j]), 0)
    #         k += 1

    return S1, S1tot  #, S2

def active_subspaces(func, k_dims, n_Ls, *args, **kwargs):
    # Set up the design:
    x_star = my_rng.random((n_Ls, k_dims))
    x_opp = my_rng.random((n_Ls, k_dims))
    # TODO: Add options for LHD, Sobol, etc.
    Δ = x_opp - x_star
    Ls = tile(x_star.reshape((n_Ls, 1, k_dims)), (1, (k_dims + 1), 1))
    for j  in range(k_dims):
        Ls[:, (j + 1), j] = x_opp[:, j]

    # Evaluate the function:
    f = func(Ls.reshape((-1, k_dims)), *args, **kwargs).reshape(Ls.shape[:2])

    # Perform the sensitivity analysis:
    d_ni = (f[:, 1:] - f[:, 0].reshape((-1, 1))) / Δ  # each of the elementary effects
    Λ, V = eigh(d_ni.T @ d_ni)
    return Λ, V


if __name__ == "__main__":
    # Linear example (trivial):
    k_dims = 5
    C = my_rng.random(1 + k_dims) - 0.5
    C[0] *= 4
    C[1:] *= 2
    print(C[1:])
    my_func = lambda x: C[0] + x @ C[1:]
    μs, μas, σs = morris(my_func, k_dims, p_disc=6, n_paths=10, campolongo=100)
    print(μs)
    μs, μas, σs = multi_L(my_func, k_dims, n_Ls=10)
    print(μs)
    S1, S1tot, S2 = sobol(my_func, k_dims, 10)
    print(sqrt(S1))
    print(S2)