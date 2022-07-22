"""
This module provides several useful utilities:
 - Use central differencing to calculate the curvature of a function at the mode
   mode in multiple dimensions — providing the covariance of a normal approximation.
 - Create a one dimensional grid that is stretched for higher density near a mode.
 - Evaluate the posterior on a mesh (a plaid grid in higher dimensions).
 - Perform approximate inverse transform sampling in multiple dimensions
   (only for dimensionality <= 8 or so depending on the expense of your posterior).
 - Perform regional sampling by correcting a normal sample (at a local mode) with
   both importance and rejection sampling.
 - Calculate a percentile from scatter data with weights.
 - Creating scatterplot matrixes and simliar array of plots for contours
   of pairwise marginals when provided a gridded distribution.
Created in June-Oct. 2019, author: Sean T. Smith
"""

from numpy import (array, empty, zeros, ones, ones_like, linspace, moveaxis, take_along_axis, meshgrid,
                   histogram2d, interp, searchsorted, prod, sqrt, exp, log)
from numpy.linalg import eigh
from numpy.random import default_rng
from scipy.special import erf, erfinv
from numba import jit, prange
from numba.extending import is_jitted

import matplotlib.pyplot as plt

my_rng = default_rng()
rand = my_rng.random
std_norm = my_rng.standard_normal

def covariance(func, x0, *args, decomp=True, **kwargs):
    """
    Use central differencing to approximate the curvature of -ln(PDF)
    (supplied func) at a mode (supplied x0) and use the curvature as the
    covariance for the multivariate normal approximation of the PDF.
    The covariance can be returned as an eigen decomposition or a matrix.
    """
    n = x0.shape[0]
    δi, δj = zeros(n), zeros(n)
    Σinv = empty((n, n))
    fmid = func(x0, *args, **kwargs)
    for i in range(n):
        δi[i] = max(1e-5 * abs(x0[i]), 1e-13)  # step size for finite diff.
        fplus  = func(x0 + δi, *args, **kwargs)
        fminus = func(x0 - δi, *args, **kwargs)
        Σinv[i, i] = (fplus - 2 * fmid + fminus) / δi[i]**2
        for j in range(i):
            δj[j] = max(1e-5 * abs(x0[j]), 1e-13)  # step for the mixed diff.
            fpp = func(x0 + δi + δj, *args, **kwargs)
            fpm = func(x0 + δi - δj, *args, **kwargs)
            fmp = func(x0 - δi + δj, *args, **kwargs)
            fmm = func(x0 - δi - δj, *args, **kwargs)
            Σinv[i, j] = Σinv[j, i] = (fpp-fpm-fmp+fmm) / (4 * δi[i] * δj[j])
            δj[j] = 0
        δi[i] = 0
    Λinv, V  = eigh(Σinv)
    if decomp:
        return 1 / Λinv, V
    else:
        return (V / Λinv) @ V.T

def normally_stretched_grid(μ, var, n, range_mult=3, clip_min=None, clip_max=None, cluster=1):
    """Create a 1D grid that is clustered in the center."""
    umin = 0.5 * (1 + erf(-cluster * range_mult / sqrt(2)))
    umax = 1 - umin
    if clip_min is not None:
        umin = max(umin, 0.5 * (1 + erf(cluster * (clip_min - μ) / (sqrt(2 * var)))))
    if clip_max is not None:
        umax = min(umax, 0.5 * (1 + erf(cluster * (clip_max - μ) / (sqrt(2 * var)))))
    uniform_grid = linspace(umin,  umax, n)
    stretched_grid  = μ + sqrt(2 * var) / cluster * erfinv(2 * uniform_grid - 1)
    return stretched_grid

@jit(nopython=True)
def func_on_1D_mesh(func, grid0, *args):
    """Calculate a function on a mesh and save the results to a 1 dimensional array."""
    x = empty(1)
    n0 = grid0.shape[0]
    output = empty(n0)
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        output[i] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_2D_mesh(func, grid0, grid1, *args):
    """Calculate a function on a mesh and save the results to a 2 dimensional array."""
    x = empty(2)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    output = empty((n0, n1))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            output[i, j] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_3D_mesh(func, grid0, grid1, grid2, *args):
    """Calculate a function on a mesh and save the results to a 3 dimensional array."""
    x = empty(3)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    output = empty((n0, n1, n2))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                output[i, j, k] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_4D_mesh(func, grid0, grid1, grid2, grid3, *args):
    """Calculate a function on a mesh and save the results to a 4 dimensional array."""
    x = empty(4)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    n3 = grid3.shape[0]
    output = empty((n0, n1, n2, n3))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                for l in range(grid3.shape[0]):
                    x[3] = grid3[l]
                    output[i, j, k, l] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_5D_mesh(func, grid0, grid1, grid2, grid3, grid4, *args):
    """Calculate a function on a mesh and save the results to a 5 dimensional array."""
    x = empty(5)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    n3 = grid3.shape[0]
    n4 = grid4.shape[0]
    output = empty((n0, n1, n2, n3, n4))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                for l in range(grid3.shape[0]):
                    x[3] = grid3[l]
                    for m in range(grid4.shape[0]):
                        x[4] = grid4[m]
                        output[i, j, k, l, m] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_6D_mesh(func, grid0, grid1, grid2, grid3, grid4, grid5, *args):
    """Calculate a function on a mesh and save the results to a 6 dimensional array."""
    x = empty(6)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    n3 = grid3.shape[0]
    n4 = grid4.shape[0]
    n5 = grid5.shape[0]
    output = empty((n0, n1, n2, n3, n4, n5))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                for l in range(grid3.shape[0]):
                    x[3] = grid3[l]
                    for m in range(grid4.shape[0]):
                        x[4] = grid4[m]
                        for n in range(grid5.shape[0]):
                            x[5] = grid5[n]
                            output[i, j, k, l, m, n] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_7D_mesh(func, grid0, grid1, grid2, grid3, grid4, grid5, grid6, *args):
    """Calculate a function on a mesh and save the results to a 7 dimensional array."""
    x = empty(7)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    n3 = grid3.shape[0]
    n4 = grid4.shape[0]
    n5 = grid5.shape[0]
    n6 = grid6.shape[0]
    output = empty((n0, n1, n2, n3, n4, n5, n6))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                for l in range(grid3.shape[0]):
                    x[3] = grid3[l]
                    for m in range(grid4.shape[0]):
                        x[4] = grid4[m]
                        for n in range(grid5.shape[0]):
                            x[5] = grid5[n]
                            for o in range(grid6.shape[0]):
                                x[6] = grid6[o]
                                output[i, j, k, l, m, n, o] = func(x, *args)
    return output

@jit(nopython=True)
def func_on_8D_mesh(func, grid0, grid1, grid2, grid3, grid4, grid5, grid6, grid7, *args):
    """Calculate a function on a mesh and save the results to a 8 dimensional array."""
    x = empty(8)
    n0 = grid0.shape[0]
    n1 = grid1.shape[0]
    n2 = grid2.shape[0]
    n3 = grid3.shape[0]
    n4 = grid4.shape[0]
    n5 = grid5.shape[0]
    n6 = grid6.shape[0]
    n7 = grid7.shape[0]
    output = empty((n0, n1, n2, n3, n4, n5, n6, n7))
    for i in range(grid0.shape[0]):
        x[0] = grid0[i]
        for j in range(grid1.shape[0]):
            x[1] = grid1[j]
            for k in range(grid2.shape[0]):
                x[2] = grid2[k]
                for l in range(grid3.shape[0]):
                    x[3] = grid3[l]
                    for m in range(grid4.shape[0]):
                        x[4] = grid4[m]
                        for n in range(grid5.shape[0]):
                            x[5] = grid5[n]
                            for o in range(grid6.shape[0]):
                                x[6] = grid6[o]
                                for p in range(grid7.shape[0]):
                                    x[7] = grid7[p]
                                    output[i, j, k, l, m, n, o, p] = func(x, *args)
    return output

# The following function generalizes the previous to nx dimensions,
#   but I could not get it to work in numba.
def func_on_mesh(func, grids, *args, ind=None, x=None, loop=0, output=None, **kwargs):
    """Calculate a function on a mesh and save the results to a multidimensional array."""
    nx = len(grids)
    if ind is None:
        ind, x = [None, ] * nx, empty(nx)
        output = empty([grid.shape[0] for grid in grids])
    if loop < nx:
        for i in range(grids[loop].shape[0]):
            ind[loop] = i
            x[loop] = grids[loop][i]
            output = func_on_mesh(func, grids, *args, ind=ind, x=x,
                                  loop=(loop + 1), output=output, **kwargs)
    else:
        output[tuple(ind)] = func(x, *args, **kwargs)
        # (Numpy requires a tuple for the index of a multidimensional array.)
    return output

def post_on_mesh(nln_post, grids, *args, **kwargs):
    """
    Calculate the posterior (given a function for the negative log. posterior, nln_post) on a
    mesh (given a list of one dimensional grids, grids), take the exponential of the negative,
    normalize using the trapezoidal rule, and return the result as the multidimensional array,
    post.
    """
    nx = len(grids)
    # Calculate the posterior on a mesh (a.k.a. a plaid grid):
    if is_jitted(nln_post):
        # Note: The optional key-word arguments are not passed for jitted functions.
        if   nx == 1:  nlnP = func_on_1D_mesh(nln_post, *grids, *args)
        elif nx == 2:  nlnP = func_on_2D_mesh(nln_post, *grids, *args)
        elif nx == 3:  nlnP = func_on_3D_mesh(nln_post, *grids, *args)
        elif nx == 4:  nlnP = func_on_4D_mesh(nln_post, *grids, *args)
        elif nx == 5:  nlnP = func_on_5D_mesh(nln_post, *grids, *args)
        elif nx == 6:  nlnP = func_on_6D_mesh(nln_post, *grids, *args)
        elif nx == 7:  nlnP = func_on_7D_mesh(nln_post, *grids, *args)
        elif nx == 8:  nlnP = func_on_8D_mesh(nln_post, *grids, *args)
        else:
            raise NotImplementedError('Capability for dimensions >= 9 has not been written!')
    else:
        nlnP = func_on_mesh(nln_post, grids, *args, **kwargs)
    post = exp(nlnP.min() - nlnP)  # Using the mode as an offset avoids overflow.

    # Integrate over the entire array:
    norm = post.copy()
    for k in range(nx):
        shape = (-1,) + (1,) * (nx - (k + 1))
        Δx = (grids[k][1:] - grids[k][:-1]).reshape(shape)
        norm = 0.5 * (Δx * norm[:-1] + Δx * norm[1:]).sum(axis=0)
    post /= norm  # normalize the posterior
    return post

def inverse_transform(pdf, x_grid, U=None, ns=100, fast=False):
    """
    Sample using approximate inverse transform sampling extended to multiple dimensions
    Which in one dimension is:
    y_grid, Δy = linspace(1.0, 5.5, 200, retstep=True)
    pdf_y = <target>(y_grid) # target pdf evaluated at y_grid
    cum_y = Δy * pdf_y.cumsum() # CDF of the target on y_grid
    n_samples = 100  # number of desired samples
    # Interpolate the inverse of the CDF
    my_rng = default_rng()
    y_samples = interp(my_rng.random(n_samples), cum_y, y_grid)
    """
    n_dim = pdf.ndim
    if U is None:
        U = rand((ns, n_dim))
    else:
        ns = U.shape[0]
    # Calculate the marginal for the 1st dimension:
    marg_x0 = pdf.copy()
    for i in range(1, n_dim):
        shape = (1, -1,) + (1,) * (n_dim - (i + 1))
        Δxi = (x_grid[i][1:] - x_grid[i][:-1]).reshape(shape)
        # trapezoid rule...
        marg_x0 = 0.5 * (Δxi*marg_x0[:,:-1] + Δxi*marg_x0[:,1:]).sum(axis=1)
    # Calculate the cumulative across the 1st dimension:
    Δx0 = x_grid[0][+1:] - x_grid[0][:-1]
    cum_x0 = empty(pdf.shape[0])
    cum_x0[0] = 0
    cum_x0[1:] = 0.5 * (Δx0 * marg_x0[:-1] + Δx0 * marg_x0[1:]).cumsum()
    cum_x0 /= cum_x0[-1]
    # Perform inverse transform sampling on the marginal:
    X = empty((ns, n_dim))
    X[:, 0] = interp(U[:, 0], cum_x0, x_grid[0])
    if n_dim > 1:
        # TODO: Optionally parallelize this loop.
        for i in range(ns):
            # Condition on sample:
            ind = searchsorted(x_grid[0], X[i, 0])
            α = ((X[i, 0]        - x_grid[0][ind-1]) /
                 (x_grid[0][ind] - x_grid[0][ind-1]))  # incorrect when ind==0
            if fast or ind == 0:
                # Nearest neighbor interpolation:
                if α <= 0.5 and ind > 0:
                    cond_pdf = pdf[ind-1]
                else:
                    cond_pdf = pdf[ind]
            else:
                # Linear interpolation:
                cond_pdf = (1 - α) * pdf[ind - 1] + α * pdf[ind]
                # This is the bottleneck for high-dims. with many samples.
            # Recurse:
            X[i, 1:] = inverse_transform(cond_pdf, x_grid[1:], U[i:i+1, 1:])
    return X

def regional_sampling(nln_target, μ, Λ, V, *args, ns0=1000, widen=1.0,
                      mins=None, maxes=None, verbose=False, **kwargs):
    """
    Sample a target distribution (the negative log. of which is provided as the function
    nln_target, within an additive const.) in the region of a previously identified mode
    (located a μ with a covariance decomposed to Λ and V).
    Careful: The returned weights are not normalized by number. Normalize manually.
    """
    nx = μ.shape[0]
    Zs = std_norm((ns0, nx))
    Xs = μ + Zs @ (V * widen * sqrt(Λ)).T

    # Enforce the mins and maxes:
    if mins is not None:
        for i in range(nx):
            if mins[i] is not None:
                Xs = Xs[Xs[:, i] >= mins[i]]
    if maxes is not None:
        for i in range(nx):
            if maxes[i] is not None:
                Xs = Xs[Xs[:, i] <= maxes[i]]
    ns1 = Xs.shape[0]
    if verbose:
        print(f'In regional_sampling, {(ns0 - ns1) / ns0 * 100:.1f}% of the points are out of '
               'bounds.')

    # Evaluate the target distribution & the sampling envelope at each sampled points:
    nlnP = empty(ns1)
    for i in range(ns1):
        nlnP[i] = nln_target(Xs[i], *args, **kwargs)
    nlnP_mode = nln_target(μ, *args, **kwargs)
    p_target = exp(nlnP_mode - nlnP)
    p_env = exp(-0.5 * (((Xs - μ) @ ((V / (widen**2 * Λ)) @ V.T)) * (Xs - μ)).sum(axis=1))

    # Calculate the importance-sampling weights:
    weights = p_target / p_env
    if verbose:
        print(f'In regional_sampling, {(weights > 1).sum() / ns0 * 100:.1f}% of the points are '
              f'out of the envelope, with a maximum weight of {weights.max():.3g}.')

    # Where weights < 1, perform rejection sampling:
    keep = rand(ns1) < weights
    ns = keep.sum()
    if verbose:
        print(f'In regional_sampling, {(ns1 - ns) / ns0 * 100:.1f}% of the points are rejected.')
        print(f'...of {ns0} initial samples, {ns} were kept.')
    Xs, weights = Xs[keep], weights[keep]
    weights = weights.clip(min=1)
    return Xs, weights

def percentile(percent, values, weights=None, axis=0, presorted=False):
    """
    Calculate the percentile boundary(s) from a set of data samples.
    This implementation is close to numpy.percentile, but supports weights.
    parameters
        percent:   float percent amount - between 0 and 100 inclusive,
        values:    array of data samples,
        weights:   1D array of weights, same length as values along the specified axis,
        axis:      integer axis of values that is traversed for computation,
        presorted: bool, if True then avoid the sorting of the initial array.
    returns
        res:  array of computed percentile boundary(s).
    Heavily modified from user Alleo's answer on the stackoverflow post:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    if weights is None:
        weights = ones(values.shape[axis])
    if not presorted:
        sorter = values.argsort(axis=axis)
        values = take_along_axis(values, sorter, axis=axis)
        weights = weights[sorter]
    data_quant = 100 * (weights.cumsum(axis=axis) - 0.5 * weights)
    data_quant /= weights.sum(axis=axis, keepdims=True)
    res_shape = [n for i, n in enumerate(values.shape) if i != axis]
    n = int(prod(res_shape))
    data_quant = moveaxis(data_quant, axis, -1).reshape((n, -1))
    values = moveaxis(values, axis, -1).reshape((n, -1))
    res = empty(n)
    for i in range(n):
        res[i] = interp(percent, data_quant[i], values[i])
    return res.reshape(res_shape)

def scatterplot_matrix(x, labels, weights=None, plot_type='scatter', ax_label_font=14,
                       fig_options={}, marginal_options={}, joint_options={}):
    """Create a scatterplot matrix from an array of samples, x."""
    ndim, nsamples = x.shape
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(ndim, ndim, sharex='col', sharey='row',
                    gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(ndim):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([percentile( 0.1, x[i], weights),
                                 percentile(99.9, x[i], weights)])
        fig.align_ylabels()
        for j in range(ndim):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([percentile( 0.1, x[j], weights),
                                  percentile(99.9, x[j], weights)])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(ndim-1):
            for j in range(i+1, ndim):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
        # Twin each diagonal axis (so the y-axis can be probability density) and format
        axes[0][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for i in range(ndim):
            ax = axes[i][i]
            twin = ax.twinx()
            axes[i][i] = (ax, twin)

    # Marginals
    nbins = max(min(nsamples // 100, 100), 20)  # This just a heuristic — adjust freely.
    for i in range(ndim):
        ax = axes[i][i][1]
        xlim = ax.get_xlim()
        bins = linspace(xlim[0], xlim[1], nbins)
        ax.hist(x[i], weights=weights, bins=bins, density=True, **marginal_options)
        ax.set_ylim([0, None])

    # Pairwise plots:
    nbins = max(min(int(sqrt(nsamples / 40)), 75), 15)  # This just a heuristic — adjust freely.
    if plot_type == 'scatter':
        # Adjust the size and/or color to reflect the weight.
        if 's' in joint_options:
            size = joint_options.pop('s')
        else:
            size = plt.rcParams['lines.markersize']
        color = None
        if weights is not None:
            size *= weights
            if 'color_by_weight' in joint_options and joint_options['color_by_weight']:
                _ = joint_options.pop('color_by_weight')
                color = log(weights)
            elif 'c' in joint_options:
                color = joint_options.pop('c')
    for i in range(ndim):
        for j in range(i):
            ax = axes[i][j]
            if plot_type == 'scatter':
                ax.scatter(x[j], x[i], s=size, c=color, **joint_options)
            elif plot_type == 'hist':
                ax.hist2d(x[j], x[i], bins=nbins, weights=weights, density=True, **joint_options)
            elif plot_type == 'contour':
                xbins = linspace(x[j].min(), x[j].max(), nbins + 1)
                ybins = linspace(x[i].min(), x[i].max(), nbins + 1)
                H, xe, ye = histogram2d(x[j], x[i], weights=weights,
                                        bins=(xbins, ybins), density=True)
                xh, yh = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
                Xh, Yh = meshgrid(xh, yh, indexing='xy')
                ax.contour(Xh, Yh, H.T, **joint_options)
    return fig, axes


def contour_matrix(pdf, x_grids, labels, plot_type='contour', ax_label_font=14,
                   fig_options={}, marginal_options={}, joint_options={}):
    """Create a contour-plot matrix from a multidimensional array of PDF values."""
    ndim = len(labels)
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(ndim, ndim, sharex='col', sharey='row',
                    gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(ndim):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([x_grids[i][0], x_grids[i][-1]])
        fig.align_ylabels()
        for j in range(ndim):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([x_grids[j][0], x_grids[j][-1]])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(ndim-1):
            for j in range(i+1, ndim):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
        # Twin each diagonal axis (so the y-axis can be probability density) and format
        axes[0][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for i in range(ndim):
            ax = axes[i][i]
            twin = ax.twinx()
            axes[i][i] = (ax, twin)

    # Marginals
    for i in range(ndim):
        marginal = pdf.copy()
        for k in range(i):
            shape = (-1,) + (1,) * (ndim - (k + 1))
            Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
            marginal = 0.5 * (Δxk * marginal[:-1] +
                              Δxk * marginal[+1:]).sum(axis=0)
        for k in range(i + 1, ndim):
            shape = (1, -1) + (1,) * (ndim - (k + 1))
            Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
            marginal = 0.5 * (Δxk * marginal[:, :-1] +
                              Δxk * marginal[:, +1:]).sum(axis=1)
        ax = axes[i][i][1]
        ax.plot(x_grids[i], marginal, **marginal_options)
        ax.set_ylim([0, None])

    # Pairwise plots:
    for i in range(ndim):
        for j in range(i):
            joint = pdf.copy()
            for k in range(j):
                shape = (-1,) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:-1] +
                               Δxk * joint[+1:]).sum(axis=0)
            for k in range(j + 1, i):
                shape = (1, -1) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:, :-1] +
                               Δxk * joint[:, +1:]).sum(axis=1)
            for k in range(i + 1, ndim):
                shape = (1, 1, -1) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:, :, :-1] +
                               Δxk * joint[:, :, +1:]).sum(axis=2)
            X1, X2 = meshgrid(x_grids[j], x_grids[i], indexing='xy')
            ax = axes[i][j]
            if plot_type == 'contour':
                ax.contour(X1, X2, joint.T, **joint_options)
            elif plot_type == 'pcolor':
                ax.pcolor(X1, X2, joint.T, shading='auto', **joint_options)
    return fig, axes


if __name__ == "__main__":
    from numpy import array, empty, arange, histogram2d, exp, log, pi as π
    from scipy.optimize import minimize
    from numba import jit
    import matplotlib.pyplot as plt

    # Define the target pdf (must be in the form of its negative log):
    @jit(nopython=True)  # This is optional — only use if you know what you're doing.
    def my_nln_pdf(y, σ2=1.0):
        x1 = log(y[0])
        x2 = y[1] - (y[0] - 2)**3 - 6
        nln_pdf = (0.5 * (((x1 - 0.5) / 0.5)**2 + ((x2 - 0) / σ2)**2) +
                   log(2 * π * 0.5 * σ2 * y[0]))
        return nln_pdf
    # print(my_nln_pdf(array([2.0, 6.0])))  # for testing

    # Find the mode and approximate the covariance at the mode:
    σ2 = 0.45  # 0.25 - 1.5
    nx = 2
    mins = [1e-5, 0]
    μ_guess = array([1.25, 6])
    out = minimize(my_nln_pdf, μ_guess, args=(σ2), method='Nelder-Mead')
    μ = out.x
    Λ, V = covariance(my_nln_pdf, μ, σ2=σ2)
    Σ = (V * Λ) @ V.T  # covariance of the normal approximation at the mode
    print('σ at the mode:')
    print(sqrt(Σ[arange(nx), arange(nx)]))

    # Create a non-uniform grid & Calculate the PDF at each point:
    n_grid = [350,  250]
    mult = [4.5, 8.5]
    cluster = [0.2, 0.3]
    grids = []
    for i in range(nx):
        grids += [normally_stretched_grid(μ[i], Σ[i, i], n_grid[i],
                  range_mult=mult[i], clip_min=mins[i], cluster=cluster[i])]
    pdf = post_on_mesh(my_nln_pdf, grids, σ2)

    # Approximate inverse transform sampling:
    Xs_it = inverse_transform(pdf, grids, ns=1200)

    # Sample in the region of the mode:
    Xs, weights = regional_sampling(my_nln_pdf, μ, Λ, V, ns0=10_000_000, widen=6.0,
                                    mins=mins, verbose=True, σ2=σ2)

    # Plot the results (multiple plot calls will be overlaid):
    names = ['$X_1$', '$X_2$']
    # ...first, from the grid
    my_fig = contour_matrix(pdf, grids, names, plot_type='pcolor',
                            fig_options=dict(figsize=(8, 7)), joint_options=dict(cmap='GnBu'))
    # ...then, contors based on all of the regional samples
    levels = pdf.max() * linspace(0.05, 0.95, 10)
    scatterplot_matrix(Xs.T, names, weights=weights, plot_type='contour',
                       fig_options=my_fig, marginal_options=dict(color='tab:blue', alpha=0.85),
                       joint_options=dict(levels=levels, linewidths=0.5, alpha=0.75))
    # # ...finally, a scatter plot for the inverse-transform samples (hide these marginals)
    # scatterplot_matrix(Xs_it.T, names, plot_type='scatter', fig_options=my_fig,
    #                    marginal_options=dict(alpha=0), joint_options=dict(s=0.5))
    # ...finally, a scatter plot for a subset of the regional samples (hide these marginals)
    n_sca = 1200
    scatterplot_matrix(Xs[:n_sca].T, names, weights=weights[:n_sca], plot_type='scatter',
                       fig_options=my_fig, marginal_options=dict(visible=False),
                       joint_options=dict(s=0.5, color_by_weight=True, alpha=0.75))
    plt.show()