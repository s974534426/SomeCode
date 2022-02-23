import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import torch
from torch.quasirandom import SobolEngine
from synthetic_function import Ackley, Hartmann


def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


np.random.seed(42)


def DropoutBO:
    def __init__(self, dims, x_min, x_max, v_min, v_max, n_init=10, seed=42):
        self.dims = dims
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.n_init = n_init
        self.seed = seed

        self.X = []
        self.new_X = []
        self.y = []
        self.g_best = None
        
    def ask(self):
        if len(self.X) == 0:
            X = latin_hypercube(self.n_init, self.dims)
            X = from_unit_cube(X, self.x_min, x_max)
            for i in range(self.n_init):
                self.X.append(X[i])
        else:
            pass

        return X

    def tell(self, fitness):
        pass


if __name__ == '__main__':
    dims = 6
    # func = Ackley(dims, True)
    func = Hartmann(dims, True)
    solver = PSO(
        dims,
        [-0] * dims,
        [1] * dims,
        [-0.1] * dims,
        [0.1] * dims
    )

    for _ in range(100):
        X = solver.ask()
        fitness = [func(x) for x in X]
        # print(fitness)
        solver.tell(fitness)
        print(solver.g_fitness)

