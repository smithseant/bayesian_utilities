"""
This module provides several useful utilities:
 - Use central differencing to calculate the curvature of a function at the mode
   mode in multiple dimensions — providing the covariance of a normal approximation.
 - Create a one dimensional grid that is stretched for higher density near a mode.
 - Evaluate the posterior on a mesh (a plaid grid in higher dimensions).
 - Perform approximate inverse transform sampling in multiple dimensions
   (only for dimensionality <= 8 or so depending on the expense of your posterior).
 - Creating scatterplot matrixes and similar array of plots for contours
   of pairwise marginals when provided a gridded distribution.
Created in June-Oct. 2019, author: Sean T. Smith
"""

from numpy import (array, empty, zeros, ones, linspace, s_, moveaxis, take_along_axis, expand_dims,
                   meshgrid, histogram2d, interp, searchsorted, prod, sqrt, exp, log)
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
    Use central differencing to approximate the inverse curvature of -ln(PDF), input func, at
    the indicated point, x0, (this inverse curvature at a mode is most often used as the
    covariance for a multivariate-normal approximation of the PDF).
    Requiring 2 * n**2 + 1 function evaluations (where n is the No. of dimensions).
    The covariance can be returned as an eigen decomposition or a matrix.
    """
    n = x0.shape[0]  # No. of dimensions
    δi, δj = zeros(n), zeros(n)  # offsets in x for individual function evaluations.
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
            Σinv[i, j] = Σinv[j, i] = (fpp - fpm - fmp + fmm) / (4 * δi[i] * δj[j])
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
    n_dims = pdf.ndim
    if U is None:
        U = rand((ns, n_dims))
    else:
        ns = U.shape[0]
    # Calculate the marginal for the 1st dimension:
    marg_x0 = pdf.copy()
    for i in range(1, n_dims):
        shape = (1, -1,) + (1,) * (n_dims - (i + 1))
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
    X = empty((ns, n_dims))
    X[:, 0] = interp(U[:, 0], cum_x0, x_grid[0])
    if n_dims > 1:
        # TODO: Optionally parallelize this loop.
        for i in range(ns):
            # Condition on sample:
            ind = min(searchsorted(x_grid[0], X[i, 0]), x_grid[0].shape[0] - 1)
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


def scatterplot_matrix(x, labels=None, weights=None, plot_type='scatter', ax_label_font=14,
                       fig_options=None, marginal_options=None, joint_options=None, grid=True,
                       clip_percentiles=(0.01, 99.99)):
    r"""
    Create a scatterplot matrix (corner/pairwise-marginal plot) from an array of samples.
    
    Return an `n_dims`-by-`n_dims` grid of axes where: the diagonal displays (weighted) marginals
    as histograms while the lower triangle displays pairwise joint marginals.
    
    Parameters
    ----------
    x : ndarray, shape(n_samples, n_dims)
        Array of sample values.
    labels : list of str, optional
        Axis labels for each parameter — default to :math:`\theta_1, \theta_2, \dots`.
    weights : ndarray, shape(n_samples), optional
        Weight for each sample.  For `plot_type="scatter"`, weights scale the marker size (& color).
    plot_type : {'scatter', 'hist', 'contour'}, optional
        Style used for the lower-triangle pairwise plots.
    ax_label_font : int, optional
        Font size for the axis labels. Defaults to `14`.
    fig_options : dict or tuple, optional
        When `type(fig_options) == dict`, keyword arguments passed to `matplotlib.pyplot.subplots`.
        When `type(fig_options) == tuple`, assumed: `(fig, axes)` (as returned by a previous call).
    marginal_options : dict, optional
        Keyword arguments passed to `matplotlib.axes.Axes.hist` for the diagonally placed marginals.
    joint_options : dict, optional
        Keyword arguments passed to the pairwise plotting routine, but augmented by two optional
        keys (for `plot_type="scatter"` and when `weights` are specified) that are handled manually:
            - `s` (float) is a base marker which is scaled by `weights`;
            - `color_by_weight` (bool)  indicates whether to set color to `log(weights)`.
    grid : bool, optional
        If `True`, apply `matplotlib.gird(True, alpha=0.2)`.
    clip_percentiles : tuple of floats, optional
        Set the axis limits according to these lower and upper percentile values.
    """
    n_samples, n_dims = x.shape
    if labels is None:
        labels = [r"$\theta_{" + f"{int(i + 1)}" + r"}$" for i in range(n_dims)]
    valid_types = ('scatter', 'hist', 'contour')
    if plot_type not in valid_types:
        raise ValueError(f"plot_type must be one of {valid_types} got {plot_type!r}")
    fig_options = dict() if fig_options is None else fig_options
    marginal_options = dict() if marginal_options is None else dict(marginal_options)
    joint_options = dict() if joint_options is None else dict(joint_options)
    lo, hi = clip_percentiles
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(n_dims, n_dims, sharex='col', sharey='row',
                                 gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(n_dims):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([percentile(lo, x[:, i], weights),
                                 percentile(hi, x[:, i], weights)])
        fig.align_ylabels()
        for j in range(n_dims):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([percentile(lo, x[:, j], weights),
                                  percentile(hi, x[:, j], weights)])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(n_dims-1):
            for j in range(i+1, n_dims):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
        # Twin each diagonal axis (so the y-axis can be probability density) and format
        axes[0][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for i in range(n_dims):
            ax = axes[i][i]
            twin = ax.twinx()
            axes[i][i] = (ax, twin)

    # Marginals
    n_bins = max(min(n_samples // 300, 100), 20)  # This just a heuristic — adjust freely.
    for i in range(n_dims):
        ax_left, ax_right = axes[i][i]
        xlim = ax_right.get_xlim()
        bins = linspace(xlim[0], xlim[1], n_bins)
        ax_right.hist(x[:, i], weights=weights, bins=bins, density=True, **marginal_options)
        ax_right.set_ylim([0, None])
        if grid:
            ax_left.set_axisbelow(True)
            ax_left.xaxis.grid(True, alpha=0.2)
            ax_right.yaxis.grid(True, alpha=0.2)

    # Pairwise plots:
    n_bins = max(min(int(sqrt(n_samples / 40)), 75), 15)  # This just a heuristic — adjust freely.
    if plot_type == 'scatter':
        # Adjust the size and/or color to reflect the weight.
        size = joint_options.pop('s', plt.rcParams['lines.markersize'])
        color = joint_options.pop('c', None)
        if weights is not None:
            size *= weights
            if joint_options.pop('color_by_weight', False):
                color = log(weights)
    for i in range(n_dims):
        for j in range(i):
            ax = axes[i][j]
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            if plot_type == 'scatter':
                ax.scatter(x[:, j], x[:, i], s=size, c=color, **joint_options)
            elif plot_type == 'hist':
                x_bins = linspace(x_lim[0], x_lim[1], n_bins + 1)
                y_bins = linspace(y_lim[0], y_lim[1], n_bins + 1)
                ax.hist2d(x[:, j], x[:, i], bins=(x_bins, y_bins), weights=weights,
                          density=True, **joint_options)
            elif plot_type == 'contour':
                x_bins = linspace(x_lim[0], x_lim[1], n_bins + 1)
                y_bins = linspace(y_lim[0], y_lim[1], n_bins + 1)

                H, xe, ye = histogram2d(x[:, j], x[:, i], weights=weights,
                                        bins=(x_bins, y_bins), density=True)
                xh, yh = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
                Xh, Yh = meshgrid(xh, yh, indexing='xy')
                ax.contour(Xh, Yh, H.T, **joint_options)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            if grid:
                ax.grid(True, alpha=0.2)
    return fig, axes


def contour_matrix(pdf, x_grids, labels=None, plot_type='contour', ax_label_font=14,
                   fig_options=None, marginal_options=None, joint_options=None, grid=True):
    r"""
    Create a contour-plot matrix (corner/pairwise-marginal plot) from a gridded probability density.
    
    Return an `n_dims`-by-`n_dims` grid of axes where: the diagonal displays (weighted) marginals
    as histograms while the lower triangle displays pairwise joint marginals.
    
    Parameters
    ----------
    pdf : ndarray
        PDF values evaluated on a plaid grid.
    x_grids : list of ndarrays
        Each of the 1D grid values — not necessarily uniformly spaced,
        `len(x_grids) == pdf.ndim`  and  `len(x_grids[i]) == pdf.shape[i]`.
    labels : list of str, optional
        Axis labels for each parameter — default to :math:`\theta_1, \theta_2, \dots`.
    plot_type : {'contour', 'pcolor'}, optional
        Style used for the lower-triangle pairwise plots.
    ax_label_font : int, optional
        Font size for the axis labels. Defaults to `14`.
    fig_options : dict or tuple, optional
        When `type(fig_options) == dict`, keyword arguments passed to `matplotlib.pyplot.subplots`.
        When `type(fig_options) == tuple`, assumed: `(fig, axes)` (as returned by a previous call).
    marginal_options : dict, optional
        Keyword arguments passed to `matplotlib.axes.Axes.plot` for the diagonally placed marginals.
    joint_options : dict, optional
        Keyword arguments passed to the pairwise plotting routine.
    grid : bool, optional
        If `True`, apply `matplotlib.gird(True, alpha=0.2)`.
    """

    def integrate_dim(f, x_grid, axis):
        """Trapazoid integration of function `f` over a dimension, given that coordinate's grid."""
        slice_lo = [s_[:]] * f.ndim
        slice_hi = [s_[:]] * f.ndim
        slice_lo[axis] = s_[:-1]
        slice_hi[axis] = s_[1:]
        Δx = x_grid[1:] - x_grid[:-1]
        Δx = expand_dims(Δx,  tuple([i for i in range(f.ndim) if i != axis]))
        return 0.5 * (Δx * (f[*slice_hi] + f[*slice_lo])).sum(axis=axis)

    def marginalize(pdf, x_grids, keep_dims):
        """Integrate `pdf` down to the axes in `keep_dims`."""
        marginal = pdf
        for axis in [i for i in reversed(range(pdf.ndim)) if i not in keep_dims]:
            marginal = integrate_dim(marginal, x_grids[axis], axis)
        return marginal

    n_dims = pdf.ndim
    if labels is None:
        labels = [r"$\theta_{" + f"{int(i + 1)}" + r"}$" for i in range(n_dims)]
    fig_options = dict() if fig_options is None else fig_options
    marginal_options = dict() if marginal_options is None else dict(marginal_options)
    joint_options = dict() if joint_options is None else dict(joint_options)
    valid_types = ('contour', 'pcolor')
    if plot_type not in valid_types:
        raise ValueError(f"plot_type must be one of {valid_types} got {plot_type!r}")
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(n_dims, n_dims, sharex='col', sharey='row',
                                 gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(n_dims):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([x_grids[i][0], x_grids[i][-1]])
        fig.align_ylabels()
        for j in range(n_dims):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([x_grids[j][0], x_grids[j][-1]])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(n_dims-1):
            for j in range(i+1, n_dims):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
        # Twin each diagonal axis (so the y-axis can be probability density) and format
        axes[0][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for i in range(n_dims):
            ax = axes[i][i]
            twin = ax.twinx()
            axes[i][i] = (ax, twin)

    # Marginals
    for i in range(n_dims):
        marginal = marginalize(pdf, x_grids, keep_dims=(i,))
        ax_left, ax_right = axes[i][i]
        ax_right.plot(x_grids[i], marginal, **marginal_options)
        ax_right.set_ylim([0, None])
        if grid:
            ax_left.set_axisbelow(True)
            ax_left.xaxis.grid(True, alpha=0.2)
            ax_right.yaxis.grid(True, alpha=0.2)

    # Pairwise plots:
    for i in range(n_dims):
        for j in range(i):
            joint = marginalize(pdf, x_grids, keep_dims=(j, i))
            X1, X2 = meshgrid(x_grids[j], x_grids[i], indexing='xy')
            ax = axes[i][j]
            if plot_type == 'contour':
                ax.contour(X1, X2, joint.T, **joint_options)
            elif plot_type == 'pcolor':
                ax.pcolor(X1, X2, joint.T, shading='auto', **joint_options)
            if grid:
                ax.grid(True, alpha=0.2)
    return fig, axes


if __name__ == "__main__":
    from numpy import array, empty, arange, histogram2d, exp, log, pi as π
    from scipy.optimize import minimize
    from numba import jit
    import matplotlib.pyplot as plt

    # Define the target pdf (must be in the form of its negative log):
    @jit(nopython=True)
    def my_nln_pdf(y, μ1=0.5, σ1=0.5, c1=2.0, μ2=0.0, σ2=1.0, c2=6.0):
        x1 = log(y[0])
        x2 = y[1] - (y[0] - c1)**3 - c2
        nln_pdf = (0.5 * (((x1 - μ1) / σ1)**2 + ((x2 - μ2) / σ2)**2) +
                   log(2 * π * σ1 * σ2 * y[0]))
        return nln_pdf
    # print(my_nln_pdf(array([2.0, 6.0])))  # for testing

    # Find the mode and approximate the covariance at the mode:
    μ1, σ1, c1 = 0.5, 0.3, 2.0
    μ1, σ2, c2 = 0.0, 0.45, 6.0
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
    mult = [5.5, 18]
    cluster = [0.1, 0.3]
    grids = []
    for i in range(nx):
        grids += [normally_stretched_grid(μ[i], Σ[i, i], n_grid[i],
                  range_mult=mult[i], clip_min=mins[i], cluster=cluster[i])]
    pdf = post_on_mesh(my_nln_pdf, grids, σ2)

    # Approximate inverse transform sampling:
    Xs_it = inverse_transform(pdf, grids, ns=1200)

    # Plot the results (multiple plot calls will be overlaid):
    # ...first, from the grid
    my_fig = contour_matrix(pdf, grids, plot_type='pcolor',
                            fig_options=dict(figsize=(8, 7)), joint_options=dict(cmap='GnBu'))

    # ...finally, a scatter plot for the inverse-transform samples (hide these marginals)
    scatterplot_matrix(Xs_it, plot_type='scatter', fig_options=my_fig,
                       marginal_options=dict(alpha=0.2),
                       joint_options=dict(c='black', s=0.25, alpha=0.5))

    plt.show()