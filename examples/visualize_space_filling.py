"""
Visualize the qualitative behavior of space-filling designs.
"""

from numpy import full, concatenate, stack, linspace, meshgrid, sqrt, log10
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt

from bayesian_utilities import scatterplot_matrix
from bayesian_utilities.space_filling import (lhs_design, support_points_design)
from distributions import distributions


def vis_2d_stratified_design(method=lhs_design, n_pnts=20, n_pnts_adtl=0, position="center"):
    """...in two dimensions."""
    plt.figure(figsize=(6, 6))

    doe0 = method(n_pnts, 2, position=position)
    plt.scatter(doe0[:, 0], doe0[:, 1], color='tab:red')

    n_pnts_tot = n_pnts
    if n_pnts_adtl > 0:
        n_pnts_tot += n_pnts_adtl
        doe1 = method(n_pnts_adtl, 2, doe0, position=position)
        plt.scatter(doe1[:, 0], doe1[:, 1], color='tab:green')

    δx = 1 / (n_pnts_tot - (position == "edges"))
    x0 = δx if position != "edges" else δx / 2
    for i in range(n_pnts_tot - 1):
        plt.plot(full(2, δx * i + x0), [0, 1], color="black", linewidth=0.5, alpha=0.5)
        plt.plot([0, 1], full(2, δx * i + x0), color="black", linewidth=0.5, alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\theta_1$', fontsize=14)
    plt.ylabel(r'$\theta_2$', fontsize=14)
    plt.title(f"{method.__name__} w/ {n_pnts} init. points & {n_pnts_adtl} adtl' points",
              fontsize=14)


def vis_2d_subsampled_design(method=support_points_design, n_pnts=20, beta=0.5, stoch_frac=0.1,
                             dist="uniform", n_prop=1000, numerically=False, **kwargs):
    """...in two dimensions."""
    plt.figure(figsize=(7, 7))

    # Proposed MC samples:
    y_prop, lnp_prop = distributions[dist]["sampler"](n_prop, **kwargs)
    plt.scatter(y_prop[:min(10_000, n_prop), 0], y_prop[:min(10_000, n_prop), 1],
                color='black', s=0.1, alpha=0.35, label='MC proposals')

    # PDF contours:
    if "eval_dist" in distributions[dist]:
        grid_shape = (250, 251)
        y_ranges = distributions[dist]["default_ranges"]
        pdf_lo = distributions[dist]["pdf_lo"]
        y1, y2 = meshgrid(linspace(*y_ranges[0], grid_shape[0]),
                          linspace(*y_ranges[1], grid_shape[1]), indexing='ij')
        y_grid = stack((y1.reshape(-1), y2.reshape(-1)), axis=-1)
        pdf_grid = distributions[dist]["eval_dist"](y_grid).reshape(grid_shape)
        contours = plt.contour(y1, y2, pdf_grid, linspace(pdf_lo, pdf_grid.max() - pdf_lo, 5),
                               linewidths=0.5, zorder=-1)

    # Space-filling design:
    y_sf, w_sf = method(n_pnts, y_prop, lnp_prop,
                        beta=beta, stoch_frac=stoch_frac, return_weights=True)

    # Scatter plot:
    if numerically:
        for i, y in enumerate(y_sf):
            plt.scatter(*y, marker=f"${i + 1}$", s=50)
    else:
        plt.scatter(y_sf[:, 0], y_sf[:, 1], color='tab:red', s=(15 * sqrt(w_sf)),
                    label=str(method.__name__))

    # ...formatting
    plt.axis('tight')
    plt.xlabel('$\theta_1$', fontsize=18)
    plt.ylabel('$\theta_2$', fontsize=18)
    plt.title(f'2D PDF ({dist})\n{n_prop:,} MC samples, {n_pnts} {method.__name__} w/ beta={beta}',
              fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    if "eval_dist" in distributions[dist]:
        h, l = contours.legend_elements()
        handles.insert(0, h[0])
        labels.insert(0, "PDF contours")
    plt.legend(handles, labels, fontsize=12)
    plt.grid(True, alpha=0.25)

    # Histogram of weights:
    if dist != "uniform" and beta > 0:
        plt.figure(figsize=(5, 5))
        plt.hist(log10(w_sf), 10, color='tab:red')
        plt.grid(True, alpha=0.2)
        plt.xlabel(r'log$_{10}$(weights) [-]  ($\Sigma_{i=1}^n w_i = n$)', fontsize=14)
        plt.ylabel('frequency [-]', fontsize=14)
        plt.title(f'range(w) = [{w_sf.min():0.3g}, {w_sf.max():0.3g}],  ' + r'ESS$_\beta$/$n$ = '
                  + f'{100 * n_pnts / (w_sf**2).sum():0.1f}%', fontsize=14)


def vis_hidim_subsampled_design(method=support_points_design, n_pnts=200, beta=0.5, stoch_frac=0.1,
                                n_prop=20_000, n_dims_uni=6, dgen_uni=2, n_dims_mvn=10, dgen_mvn=3):
    """...dimensionality >=4."""

    # Proposed MC samples:
    y_uni, lnp_uni = distributions["uniform"]["sampler"](n_prop, n_dims_uni, False, dgen_uni)
    y_mvn, lnp_mvn = distributions["multivar. normal"]["sampler"](n_prop, n_dims_mvn, dgen_mvn)
    y_wisp, lnp_wisp = distributions["blob w/ a wisp"]["sampler"](n_prop)
    y_cube, lnp_cube = distributions["cubic manifold"]["sampler"](n_prop)

    y_prop = concatenate((y_uni, y_mvn, y_wisp, y_cube), axis=1)
    lnp_prop = lnp_uni + lnp_mvn + lnp_wisp + lnp_cube

    # Space-filling design:
    y_sf, w_sf = method(n_pnts, y_prop, lnp_prop,
                        beta=beta, stoch_frac=stoch_frac, return_weights=True)

    # Scatter-plot matrix:
    spm = scatterplot_matrix(y_prop[:min(n_prop, 5_000)], fig_options=dict(figsize=(15, 14)),
                                joint_options=dict(s=0.1, c='black', alpha=0.25),
                                marginal_options=dict(color='black'))
    if beta == 0:
        spm = scatterplot_matrix(y_sf, fig_options=spm, joint_options=dict(c='tab:red', s=50))
    else:
        spm = scatterplot_matrix(y_sf, weights=w_sf, fig_options=spm,
                                 joint_options=dict(c='tab:red', s=0.25 * sqrt(w_sf)),
                                 marginal_options=dict(color='tab:red', alpha=0.5))

    plt.suptitle((f'{n_dims_uni + n_dims_mvn + 4} dimensional PDF:  '
                  f'{n_dims_uni} ' + r'$\mathcal{U}$'
                  f' + {n_dims_mvn} ' + r'$\mathcal{N}$'
                  f' +  2 blob w/ a wisp + 2 cubic manifold'
                  f'\n{n_prop:,} MC proposals, {n_pnts} {method.__name__} w/ beta={beta}'),
                 fontsize=14)

    # Emphasize pairwise marginal (of the blob w/ a wisp):
    for i, dist, ym_prop in zip((16, 18), ("blob w/ a wisp", "cubic manifold"), (y_wisp, y_cube)):
        plt.figure(figsize=(7, 7))
        # PDF contours...
        grid_shape = (250, 251)
        y_ranges = distributions[dist]["default_ranges"]
        y1, y2 = meshgrid(linspace(*y_ranges[0], grid_shape[0]),
                            linspace(*y_ranges[1], grid_shape[1]), indexing='ij')
        y_grid = stack((y1.reshape(-1), y2.reshape(-1)), axis=-1)
        pdf_grid = distributions[dist]["eval_dist"](y_grid).reshape(grid_shape)
        pdf_lo = distributions[dist]["pdf_lo"]
        con = plt.contour(y1, y2, pdf_grid, linspace(pdf_lo, pdf_grid.max() - pdf_lo, 5),
                          linewidths=0.5, zorder=-1)
        # plot the marginal of the proposal points...
        plt.scatter(ym_prop[:min(10_000, n_prop), 0], ym_prop[:min(10_000, n_prop), 1],
                    color='black', s=0.1, alpha=0.35, label='MC samples')
        # plot the marginal of the space-filling points...
        plt.scatter(y_sf[:, i], y_sf[:, i + 1], color='tab:red', s=(10 * sqrt(w_sf)),
                    label=method.__name__)
        # ...formatting
        plt.xlim(y_ranges[0])
        plt.ylim(y_ranges[1])
        plt.xlabel(r'$\theta_{$' + f'{i}' + r'}$', fontsize=18)
        plt.ylabel(r'$\theta_{$' + f'{i + 1}' + r'}$', fontsize=18)
        plt.title((f'{n_dims_uni + n_dims_mvn + 4} dimensional PDF — '
                   f'marginal of {dist}'
                   f'\n{n_prop:,} MC samples, {n_pnts} {method.__name__} w/ beta={beta}'),
                  fontsize=14)
        handles, labels = plt.gca().get_legend_handles_labels()
        h, l = con.legend_elements()
        handles.insert(0, h[0])
        labels.insert(0, "PDF contours")
        plt.legend(handles, labels, fontsize=12)
        plt.grid(True, alpha=0.25)


if __name__ ==  "__main__":
    vis_2d_stratified_design(n_pnts=6, n_pnts_adtl=24)
    vis_2d_subsampled_design(dist="uniform", numerically=True)
    vis_2d_subsampled_design(dist="blob w/ a wisp")
    vis_hidim_subsampled_design()
    plt.show()