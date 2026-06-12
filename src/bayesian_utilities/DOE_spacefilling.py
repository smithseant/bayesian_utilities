# -*- coding: utf-8 -*-
"""
This is a simple toolkit to perform design-of-experiment sampling according
to my preferred methods - Latin hyper-cube sampling & optimized design
based on a potential field.

A Sobol sequence is another popular & legitimate approach, however
implementation is much more involved and is not available in numpy or scipy.
One suggested library can be found at:  https://github.com/SALib/SALib.

Created April 2017 @author: Sean T. Smith
"""
from bisect import bisect_left
from numpy import empty, zeros, ones, concatenate, amin, abs, sqrt
from numpy.random import random_sample, permutation
from scipy.special import erf, erfinv
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt

class DOE:
    """
    Base class for a design of experiments. Provides structure and utilities,
    including the distance calculation, identification of the minimum distance,
    and a plotting tool.
    """
    def __init__(self, n_pts, n_dims, method, *args,
                 scale=None, fixed_pts=None, **kwargs):
        self.n_pts = n_pts
        self.n_fixed = 0 if fixed_pts is None else fixed_pts.shape[0]
        self.n_dims = n_dims
        self.scale = ones(n_dims) if scale is None else scale
        self.x_fixed = fixed_pts
        self.x = method(n_pts, n_dims, *args,
                        fixed_pts=fixed_pts, **kwargs)
        if fixed_pts is None:
            D = self.distance(self.x, self.x, scale=self.scale)
        else:
            x_combined = concatenate((self.x, fixed_pts))
            D = self.distance(x_combined, self.x, scale=self.scale)
        self.δ, self.nδ = DOE.min_dist(D)

    def u2z(self, *args, **kwargs):
        self.x = u2z(self.x, *args, **kwargs)
        return None

    def z2u(self, *args, **kwargs):
        self.x = z2u(self.x, *args, **kwargs)
        return None

    @staticmethod
    @jit(nopython=True)
    def distance(x, y, p=2, scale=None, root=True):
        """
        Calculate the distance (Lp norm) between the points in x & y.
        Arguments
            x:  ndarray-2D,
                The first array of points with 1st dim. for the individual
                points, and the 2nd dim. for the coordinate directions.
            y:  ndarray-2D,
                The second array of points - same format.
            p:  positive integer (default: 2),
                The order of norm to be calculated.
            scale: None or array-1D,
                A 1D-array with length of the dimensionality of x & y used
                to scale the distance in each direction.
            root: bool,
                If True - return the rooted p norm,
                if False - return the unrooted sum.
        Returns
            Lp:  ndarray-2D,
                The returned norm is an array (with each dim. corresponding
                the input points).
        """
        nx_pts, n_dims = x.shape
        ny_pts, n_dims = y.shape
        if scale is None:
            s = ones(n_dims)
        else:
            s = scale
        Lp = empty((nx_pts, ny_pts))
        p_inv = 1.0 / p
        for i in range(nx_pts):
            for j in range(ny_pts):
                my_sum = 0.0
                for k in range(n_dims):
                    my_sum += abs((x[i, k] - y[j, k]) / s[k])**p
                Lp[i, j] = my_sum
        if root:
            for i in range(nx_pts):
                for j in range(ny_pts):
                    Lp[i, j] = Lp[i, j] ** p_inv
        return Lp

    @staticmethod
    @jit(nopython=True)
    def min_dist(L):
        """
        Identify the minimum argument over the lower triangular portion
        of a square input matrix, L, and return the number of entries
        with that same minimum value.
        """
        δmin = L[1, 0] + 1
        count = 1
        for i in range(L.shape[0]):
            for j in range(min(i, L.shape[1])):
                if L[i, j] < δmin:
                    δmin = L[i, j]
                    count = 1
                elif L[i, j] == δmin:
                    count += 1
        return δmin, count

    @staticmethod
    def better_than(a, c):
        return a.δ > c.δ or a.δ == c.δ and a.nδ < c.nδ

    def plot(self, title=None, position=False):
        # Plot the design itself:
        if position == 'center' or position == 'random':
            dx = 1 / (self.n_pts + self.n_fixed)
            x0 = 0.5 * dx
        elif position == 'edges':
            dx = 1 / (self.n_pts + self.n_fixed - 1)
            x0 = 0
        fig_design = plt.figure()
        n_fig = 1
        for i in range(1, self.n_dims):
            for j in range(i):
                fig_design.add_subplot(self.n_dims - 1, self.n_dims - 1, n_fig)
                if position:
                    for k in range(self.n_pts + self.n_fixed - 1):
                        xn = x0 + 0.5 * dx + k * dx
                        plt.plot([0, 1], [xn, xn], color=[1.0, 0.75, 0.75],
                                 linewidth=0.5)
                        plt.plot([xn, xn], [0, 1], color=[1.0, 0.75, 0.75],
                                 linewidth=0.5)
                plt.scatter(self.x[:, j], self.x[:, i])
                if self.x_fixed is not None:
                    plt.scatter(self.x_fixed[:, j], self.x_fixed[:, i])
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                n_fig += 1
            n_fig += self.n_dims - (i + 1)
        if title is not None:
            plt.title(title)
        # Plot the radial distribution function of the design:
        fig_rdf = plt.figure()
        dist = self.distance(self.x, self.x)
        rdf = plt.hist(dist.reshape(-1), bins=40, density=True)
        plt.xlabel('Distance between pairs of points')
        plt.ylabel('Frequency')
        plt.title(title)
        return fig_design, fig_rdf


class Random(DOE):
    """
    A design of experiments that is generated by random sampling.
    """
    def __init__(self, n_pts, n_dims, *args, scale=None, **kwargs):
        # It is implicit within random sampling to ignore existing points.
        super().__init__(n_pts, n_dims, Random.random, *args,
                         scale=scale, **kwargs)

    @staticmethod
    def random(n_pts, n_dims, *args, **kwargs):
        return random_sample((n_pts, n_dims))


class LHS(DOE):
    """
    A design of experiments that is generated by latin hypercube sampling.
    """
    def __init__(self, n_pts, n_dims, *args,
                 scale=None, fixed_pts=None, **kwargs):
        super().__init__(n_pts, n_dims, LHS.latin_hypercube, *args,
                         scale=scale, fixed_pts=fixed_pts, **kwargs)

    @staticmethod
    def latin_hypercube(n_pts, n_dims, *args,
                        position='center', fixed_pts=None, **kwargs):
        """
        Single sample of a latin-hypercube design over a unit box with
        n_points in n_dims dimensions. The position of the point within
        the sub-region can be specified as 'center' (default), 'edges',
        or 'random' (with a uniform distribution).
        This method preserves the constant 1-point density of the uniform
        distribution, but not the 2-point Poisson statistics (in an attempt
        to be more 'space filling'). Has the ability to fill in around an
        optional set of fixed input points.
        """
        if fixed_pts is None:
            n_fixed = 0
        else:
            fixed = fixed_pts.copy()
            fixed = fixed[(fixed >= 0).all(axis=1)]  # Remove fixed points...
            fixed = fixed[(fixed <= 1).all(axis=1)]  # outside the unit cube.
            n_fixed = fixed.shape[0]
        if position == 'center' or position == 'random':
            dx = 1 / (n_pts  )# + n_fixed)  # test
            x0 = 0.5 * dx
        elif position == 'edges':
            dx = 1 / (n_pts   -1)# + n_fixed - 1)  # test
            x0 = 0.0
        i_position = empty((n_pts, n_dims))
        for j in range(n_dims):
            # First, list all tiles by index
            tiles = [i for i in range(n_pts  )]# + n_fixed)]  # test
            # # Second, identify & remove the tiles that contain fixed_pts
            # for i in range(n_fixed):
            #     tfixed = int(fixed[i, j] / dx)  # rounds toward zero
            #     ind = bisect_left(tiles, tfixed)
            #     if ind == 0:
            #         tiles.pop(ind)
            #     elif ind == len(tiles):
            #         tiles.pop(ind - 1)
            #     else:
            #         # removes nearest (in case multiple fixed_pts in a tile)
            #         tile_left  = tiles[ind - 1]
            #         tile_right = tiles[ind]
            #         if tfixed - tile_left < tile_right - tfixed:
            #             tiles.pop(ind - 1)
            #         else:
            #             tiles.pop(ind)  # test
            # Then, permute the tiles:
            i_position[:, j] = permutation(tiles)
        lhd = x0 + dx * i_position.astype(float)
        if position == 'random':
            lhd += dx * (random_sample((n_pts, n_dims)) - 0.5)
        return lhd


class Maximin(DOE):
    def __init__(self, n_pts, n_dims, *args, method=LHS, n_samples=1000,
                 scale=None, fixed_pts=None, verbose=True, **kwargs):
        """
        Naively generate multiple designs and return the one that
        best satisfies the Morris & Mitchell (1995) maximin criterion.
        Arguments
            n_points: positive integer,
                the number of points in each design.
            n_dims:  positive integer,
                the number of dimensions in the design space.
            method:  DOE (default: LHD),
                the method used for creation of a new design.
            n_samples:  positive integer (default: 1000),
                the number of unique designs in the optimization procedure.
            scale:  None or array-1D (default: None),
                optional array for scaling each dimension separately.
            fixed_pts:  None or array-2D (default: None),
                optional additional points that have a fixed position.
            verbose:  bool (default: True),
                if true, print progress and result.
        """
        best = method(n_pts, n_dims, *args,
                      scale=scale, fixed_pts=fixed_pts, **kwargs)
        if verbose:
            print('Comparing {:d} designs...'.format(n_samples))
        for i in range(1, n_samples):
            # Generate a candidate design and compare to the previous best.
            new = method(n_pts, n_dims, method, *args,
                         scale=scale, fixed_pts=fixed_pts, **kwargs)
            # Compare using the Morris & Mitchell maximin criterion.
            if DOE.better_than(new, best):
                if verbose:
                    print('   Design {:d} has been accepted!'.format(i + 1))
                best = new
        self.__dict__.update(best.__dict__)

class PotentialField(DOE):
    def __init__(self, doe, verbose=True):
        """
        Creates an experimental design by minimizing a potential field between
        points. The potential field between points is defined by the overlap of
        space close to points. The solution algorithm moves points toward the
        local minimum of potential.
        Careful: this algorithm depends on the initial design, doe.
        """
        bnds = tuple((0.0, 1.0) for i in range(doe.n_pts * doe.n_dims))
        solution = minimize(PotentialField.objective, doe.x.reshape(-1),
                      args=(doe.n_pts, doe.n_dims, doe.scale, doe.x_fixed),
                      method='L-BFGS-B', jac=True, bounds=bnds)
        if verbose:
            print('{} function evaluations'.format(solution.nfev))
            print('{} interations'.format(solution.nit))
        self.__dict__.update(doe.__dict__)
        self.x = solution.x.reshape((doe.n_pts, doe.n_dims))

    # @jit(nopython=True)
    @staticmethod
    def potential(x, scale, fixed_pts=None):
        f_wall = 2.0  # Factor for the wall-potential magnitude
        np_wall = 0.01  # Parameter for wall force drop-off
        n_pts, n_dims = x.shape
        dist = DOE.distance(x, x, p=2, scale=scale, root=False) + 1e-8
        if fixed_pts is None:
            n_fixed = 0
        else:
            n_fixed = fixed_pts.shape[0]
            dfix = DOE.distance(x, fixed_pts, p=2, scale=scale, root=False) + 1e-8
        pot = 0.0
        force = zeros((n_pts, n_dims))  # negative gradient of potential
        for i in range(n_pts):
            # Sum of forces on point i from all previously placed points:
            for j in range(i):
                pot += 1 / dist[i, j]
            for k in range(n_dims):
                sum = 0.0
                for j in range(n_pts):
                    if not i == j:
                        sum += (x[i, k] - x[j, k]) / dist[i, j]**2
                force[i, k] += 2 * sum / scale[k]
            # Sum forces on point i from all the fixed points:
            for j in range(n_fixed):
                pot += 1 / dfix[i, j]
            for k in range(n_dims):
                sum = 0.0
                for j in range(n_fixed):
                    sum += (x[i, k] - fixed_pts[j, k]) / dfix[i, j]**2
                force[i, k] += 2 * sum / scale[k]
            # Potential with the walls:
            # ...lower walls
            for k in range(n_dims):
                e = f_wall * ((np_wall * scale[k]) / (x[i, k] - (0-1e-8)))**2
                force[i, k] += 2 * e / (x[i, k] - (0-1e-8))
                pot += e
            # ...upper walls
            for k in range(n_dims):
                e = f_wall * ((np_wall * scale[k]) / (x[i, k] - (1+1e-8)))**2
                force[i, k] += 2 * e / (x[i, k] - (1+1e-8))
                pot += e
        return pot, force

    @staticmethod
    def objective(x, n_pts, n_dims, scale, fixed_pts):
        xr = x.reshape((n_pts, n_dims))
        pot, force = PotentialField.potential(xr, scale, fixed_pts)
        return pot, -force.reshape(-1)


def u2z(u, μ=None, L=None):
    z = sqrt(2) * erfinv(2 * u - 1)
    if L is not None:
        z = z @ (L[1] * sqrt(L[0])).T
    if μ is not None:
        z += μ
    return z

def z2u(z, μ=None, L=None):
    if μ is not None:
        z -= μ
    if L is not None:
        z = z @ (L[1] / sqrt(L[0]))
    return (1 + erf(z / sqrt(2))) / 2


if __name__ == '__main__':
    from numpy import array
    from numpy.linalg import eigh

    n_pts = 4
    n_dims = 2
    position = 'random'
    scale = array([0.5, 2.0])

    lhs0 = LHS(n_pts, n_dims, scale=scale, position=position)
    dist = DOE.distance(lhs0.x, lhs0.x, scale=scale)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in initial design: {:f}'.format(min_dist))
    lhs0.plot(title='LHD (single sample)', position=position)

    lhs = Maximin(n_pts, n_dims, LHS, n_samples=int(1e4),
                  position=position, scale=scale)
    dist = DOE.distance(lhs.x, lhs.x, scale=scale)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in selected design: {:f}'.format(min_dist))
    lhs.plot(title='LHD (multiple samples)', position=position)

    pfd = PotentialField(lhs0)
    dist = DOE.distance(pfd.x, pfd.x, scale=scale)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in pot. field design: {:f}'.format(min_dist))
    pfd.plot(title='potential field design')
    print(' ')

    print('With fixed points...')
    fixed_pts = array([[0.1, 0.1], [0.7, 0.5], [0.3, 0.7]])

    lhs1 = LHS(n_pts, n_dims, fixed_pts=fixed_pts, position=position)
    dist = DOE.distance(concatenate((lhs1.x, fixed_pts)), lhs1.x)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in fixed-point design: {:f}'.format(min_dist))
    lhs1.plot(title='LHD (single sample - w/ fixed)', position=position)

    lhs = Maximin(n_pts, n_dims, LHS, n_samples=int(1e4),
                     fixed_pts=fixed_pts, position=position)
    dist = DOE.distance(concatenate((lhs.x, fixed_pts)), lhs.x)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in selected design: {:f}'.format(min_dist))
    lhs.plot(title='LHD (multiple samples - w/ fixed)', position=position)

    pfd = PotentialField(lhs1)
    dist = DOE.distance(concatenate((pfd.x, fixed_pts)), pfd.x)
    min_dist = amin(dist[dist > 0.0])
    print('Minimum distance in pot. field design: {:f}'.format(min_dist))
    pfd.plot(title='potential field design (w/ fixed)')
    print(' ')


    n_pts = 50
    n_dims = 2

    U = LHS(n_pts, n_dims).x
    Z = u2z(U)
    μ = array([2, 6])
    Σ = array([[1, 0.8], [0.8, 1]])
    Λ, V = eigh(Σ)
    X = u2z(U, μ=μ, L=(Λ, V))
    U2 = z2u(X, μ=μ, L=(Λ, V))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(U[:, 0], U[:, 1], label='original points')
    plt.scatter(U2[:, 0], U2[:, 1], label='round-trip points')
    plt.title('Uniform Sample')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.title('Standard Normal Sample')
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Normal Sample')


    plt.show()
