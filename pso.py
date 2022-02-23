import numpy as np 
import matplotlib.pyplot as plt
from synthetic_function import Ackley, Hartmann


np.random.seed(42)


class PSO:
    def __init__(self, dims, x_min, x_max, v_min, v_max, pop_size=20, seed=42):
        self.dims = dims
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.pop_size = pop_size
        self.seed = seed

        self.w = 0.9
        self.c1 = 2
        self.c2 = 2

        self.X = []
        self.V = []
        self.p_best = []
        self.p_fitness = []
        self.g_best = None
        self.g_fitness = -99999

    def _init_population(self):
        # X = latin_hypercube(self.pop_size, self.dims)
        # X = from_unit_cube(X, np.array(self.x_min), np.array(self.x_max))

        for i in range(self.pop_size):
            x = np.random.uniform(self.x_min, self.x_max)
            v = np.random.uniform(self.v_min, self.v_max)
            # self.X.append(X[i])
            self.X.append(x)
            self.V.append(v)

    def ask(self):
        if len(self.X) == 0:
            self._init_population()
        else:
            for i in range(self.pop_size):
                self.V[i] = self.w * self.V[i] + \
                    self.c1 * np.random.uniform() * (self.p_best[i] - self.X[i]) + \
                    self.c2 * np.random.uniform() * (self.g_best - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = np.clip(self.X[i], self.x_min, self.x_max)

        return self.X

    def tell(self, fitness):
        assert len(fitness) == len(self.X)
        if len(self.p_best) == 0:
            for i in range(self.pop_size):
                self.p_best.append(self.X[i])
                self.p_fitness.append(fitness[i])
                if self.p_fitness[-1] > self.g_fitness:
                    self.g_best = self.p_best[-1]
                    self.g_fitness = self.p_fitness[-1]
        else:
            for i in range(self.pop_size):
                if fitness[i] > self.p_fitness[i]:
                    self.p_best[i] = self.X[i]
                    self.p_fitness[i] = fitness[i]
                if self.p_fitness[i] > self.g_fitness:
                    self.g_best = self.p_best[i]
                    self.g_fitness = self.p_fitness[i]

        
if __name__ == '__main__':
    dims = 500
    func = Ackley(dims, True)
    # func = Hartmann(dims, True)
    solver = PSO(
        dims,
        [-10] * dims,
        [10] * dims,
        [-0.1] * dims,
        [0.1] * dims
    )
    gbest_y_hist = []

    for _ in range(100):
        X = solver.ask()
        fitness = [func(x) for x in X]
        # print(fitness)
        solver.tell(fitness)
        print(solver.g_fitness)
        gbest_y_hist.append(-solver.g_fitness)

    plt.plot(gbest_y_hist)
    plt.show()

    # from sko.PSO import PSO
    # pso = PSO(func=Ackley(dims, False), dim=dims, lb=[-10]*dims, ub=[10]*dims, pop=20, max_iter=100)
    # fitness = pso.run()
    # print(pso.gbest_y_hist)
    
    # plt.plot(pso.gbest_y_hist)
    # plt.show()
