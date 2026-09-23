"""
Reusable PDFs for examples & testing.
"""

from numpy import array, full, nan_to_num, diag, sign, sqrt, exp, log, sin, cos, pi as π
from numpy.linalg import det, inv, qr
from numpy.random import default_rng


def uniform_randshape(n_samples, n_dims=2, std=True, degeneracy=0, rng=None):
    """Sample from a uniform distribution (w/ corresponding log prob. values) using a radom box."""
    rng = default_rng() if rng is None else rng
    if std:
        m, w = 0.5, full(n_dims, 1.0)
    else:
        # mean...
        m = rng.standard_normal(n_dims)
        # widths...
        w = rng.exponential(scale=1, size=n_dims)
    w[rng.choice(n_dims, degeneracy)] = 0
    # samples & log probabilities...
    Ys = w * (rng.random((n_samples, n_dims)) - 0.5) + m
    lnPs = full(n_samples, 1.0)
    return Ys, lnPs


def multivariate_normal_randshape(n_samples, n_dims=2, degeneracy=0, rng=None):
    """Sample from a multivariate normal distribution (w/ log prob.) using random location & cov."""
    rng = default_rng() if rng is None else rng
    # mean...
    μ = rng.standard_normal(n_dims)
    # covariance...
    Λ = rng.exponential(scale=1, size=n_dims)
    Λ[:degeneracy] = 0
    Q, R = qr(rng.standard_normal((n_dims, n_dims)))
    Q *= sign(diag(R))
    Σ = (Q * Λ) @ Q.T
    # samples...
    Ys = rng.standard_normal((n_samples, n_dims)) @ (Q * sqrt(Λ)).T  + μ
    k_nd = Λ > 1e-9 * Λ.max()
    z = (Ys - μ) @ Q[:, k_nd]
    lnPs = -0.5 * (((z**2) / Λ[k_nd]).sum(axis=1) + (k_nd).sum() * log(2 * π) + log(Λ[k_nd]).sum())
    return Ys, lnPs



def blob_with_wisp(n_samples, μ1=3.0, σ1=0.6, μ2=3.0, σ2=2.0, c=8e-3, φ=π/3, rng=None):
    """Sample from a 2 dimensional PDF (w/ log prob.) that is best described as a blob w/ a wisp."""
    rng = default_rng() if rng is None else rng
    # rotation matrix...
    A = array([[ cos(φ), -sin(φ)],
               [ sin(φ),  cos(φ)]])
    # independent normals...
    Xs = array([μ1, μ2]) + rng.standard_normal((n_samples, 2)) * array([σ1, σ2])
    # transform to sample values...
    Ys_prime = array([(ys1:=exp(Xs[:, 0])),
                      Xs[:, 1] + c * ys1**2]).T
    Ys = Ys_prime @ A.T
    # sample log probabilities...
    lnPs = (-0.5 * (((Xs[:, 0] - μ1) / σ1)**2 + ((Xs[:, 1] - μ2) / σ2)**2)
            - log(2 * π * σ1 * σ2 * abs(det(A))  * Ys_prime[:, 0]))
    return Ys, lnPs


def blob_with_wisp_pdf(y, μ1=3.0, σ1=0.6, μ2=3.0, σ2=2.0, c=8e-3, φ=π/3):
    """Evaluate the blob-with-wisp PDF at coordinate values `y` (axes by the stats. convention)."""
    # rotation matrix...
    A = array([[ cos(φ), -sin(φ)],
               [ sin(φ),  cos(φ)]])
    # invert the rotation, then invert the (log & quadratic) transformations...
    y_prime = y @ inv(A).T
    x1 = log(y_prime[:, 0])
    x2 = y_prime[:, 1] - c * y_prime[:, 0]**2
    pdf = (exp(-0.5 * (((x1 - μ1) / σ1)**2 + ((x2 - μ2) / σ2)**2))
           / (2 * π * σ1 * σ2 * abs(det(A)) * y_prime[:, 0]))
    return nan_to_num(pdf, 0.0)


def cubic_manifold(n_samples, μ1=0.5, σ1=0.3, μ2=0.0, σ2=0.45, c1=2.0, c2=6.0, rng=None):
    """Sample from a 2 dimensional PDF (w/ log prob.) described as reducing to a cubic manifold."""
    rng = default_rng() if rng is None else rng
    Xs = array([μ1, μ2]) + rng.standard_normal((n_samples, 2)) * array([σ1, σ2])
    Ys = array([(y1s:=exp(Xs[:, 0])),
                (y1s - c1)**3 + c2 + Xs[:, 1]]).T
    lnPs = (-0.5 * (((Xs[:, 0] - μ1) / σ1)**2 + ((Xs[:, 1] - μ2) / σ2)**2)
            - log(2 * π * σ1 * σ2 * Ys[:, 0]))
    return Ys, lnPs


def cubic_manifold_pdf(y, μ1=0.5, σ1=0.3, μ2=0.0, σ2=0.45, c1=2.0, c2=6.0):
    """Evaluate the cubic-manifold PDF at coordinate values `y` (axes by the stats. convention)."""
    x1 = log(y[:, 0])
    x2 = y[:, 1] - (y[:, 0] - c1)**3 - c2
    pdf = (exp(-0.5 * (((x1 - μ1) / σ1)**2 + ((x2 - μ2) / σ2)**2))
           / (2 * π * σ1 * σ2 * y[:, 0]))
    return nan_to_num(pdf, nan=0.0)


distributions = {"uniform" : dict(sampler=uniform_randshape),
                 "multivar. normal" : dict(sampler=multivariate_normal_randshape),
                 "blob w/ a wisp" : dict(sampler=blob_with_wisp, eval_dist=blob_with_wisp_pdf,
                                         default_ranges=((-13, 13), (1.1, 115)), pdf_lo=2e-4),
                 "cubic manifold" : dict(sampler=cubic_manifold, eval_dist=cubic_manifold_pdf,
                                         default_ranges=((0.4, 4.0), (1.0, 13)), pdf_lo=2e-2)}