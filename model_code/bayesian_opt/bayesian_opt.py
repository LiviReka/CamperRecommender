import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import warnings
from scipy.optimize import minimize


def se(y, yhat):
    """Computing Squared Error"""
    return np.round((y - yhat) ** 2, 3)


class BayesianOpt:
    def __init__(self, X, model, exploration, objective_f, objective_bounds, n_target_obs):
        self.objective_f = objective_f

        self.X = X.reshape(1, -1)  # Initialized x values
        self.y = self._objective(X)  # objective function evaluates init xs to ys

        self.n_target_obs = n_target_obs  # target number of observations to evaluate on
        self.surr_model = model  # surrogate model to compute posterior
        self.exploration = exploration
        self.iteration = 0
        self.objective_bounds = objective_bounds

    def _objective(self, x):
        try:
            res = np.array([self.objective_f(i) for i in x])
        except TypeError:
            res = self.objective_f(x)
        return res

    def _surrogate(self, x):
        """The surrogate function describes our current knowledge of the objective"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.surr_model.predict(x, return_std=True)

    def _acquisition(self, x_new):
        """The acquisition function evaluates which sample area is to be explored next: Random Choice"""
        return -np.random.rand(1)

    def optimize(self):
        self.surr_model.fit(self.X.T, self.y)
        xs = []
        ys = []
        for i in range(self.n_target_obs):
            self.iteration += 1
            n_candidates = 2
            x_candidates = np.random.randint(low=50, high=2000, size=n_candidates).reshape(1, -1)

            new_x = None
            best = np.inf
            for j in range(len(x_candidates)):
                x_cand = x_candidates[:, j].reshape(-1, 1)
                res = minimize(self._acquisition, x_cand, method='L-BFGS-B', bounds=self.objective_bounds)
                if res.success and res.fun < best:
                    best = res.fun
                    new_x = res.x.astype('int')
                if not res.success:
                    new_x = self.X[:, -1] + 1e-1

            new_y = self._objective(new_x)

            xs.append(new_x)
            ys.append(new_y)

            self.X = np.concatenate((self.X, new_x.reshape(-1, 1)), axis=1)
            self.y = np.concatenate((self.y, new_y))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.surr_model.fit(self.X.T, self.y)

            print(f'#### OPTIMIZATION PROGRESS: {i / self.n_target_obs*100}%')
        best = np.argmin(self.y)
        return (self.X[:, best], self.y[best]), xs, ys


class BayesianOptPI(BayesianOpt):
    def __init__(self, X, model, exploration, objective_f, objective_bounds, n_target_obs=100):
        super().__init__(X, model, exploration, objective_f, objective_bounds)

    def _acquisition(self, x_new):
        """The acquisition function evaluates which sample area is to be explored next: probability of improvement"""

        yhat, _ = self._surrogate(self.X.T)
        best = np.min(yhat)

        mu, std = self._surrogate(np.expand_dims(x_new, axis=0))

        PI = norm.cdf((mu - best - self.exploration) / (std + 1e-15))
        return -PI


class BayesianOptUCB(BayesianOpt):
    def __init__(self, X, model, exploration, objective_f, objective_bounds, n_target_obs=100):
        super().__init__(X, model, exploration, objective_f, objective_bounds)

    def _acquisition(self, x_new):
        """The acquisition function evaluates which sample area is to be explored next: UCB"""
        mu, std = self._surrogate(np.expand_dims(x_new, axis=0))

        k = self.exploration * np.sqrt((2*np.log((self.iteration**(2/2. + 2))*(np.pi**2)/(3. * 1e-5))))
        return mu + k * std


class BayesianOptEI(BayesianOpt):
    def __init__(self, X, model, exploration, objective_f, objective_bounds, n_target_obs):
        super().__init__(X, model, exploration, objective_f, objective_bounds, n_target_obs)

    def _acquisition(self, x_new):
        """The acquisition function evaluates which sample area is to be explored next: UCB"""
        yhat, _ = self._surrogate(self.X.T)
        best = np.min(yhat)

        mu, std = self._surrogate(np.expand_dims(x_new, axis=0))

        k = (mu - best - self.exploration) / (std + 1e-15)

        return ((mu - best - self.exploration) * norm.cdf(k) + std * norm.pdf(k)).reshape(1, )


def branin_xs(n):
    """Randomly Drawing Xs to evaluate Branin Function on"""
    x1 = np.random.uniform(low=-5, high=10, size=n)
    x2 = np.random.uniform(low=0, high=15, size=n)
    return np.array([x1, x2])


if __name__ == '__main__':
    # Generate X Values
    X = branin_xs(5)

    model = BayesianOptEI(X=X, model=GaussianProcessRegressor(kernel=RBF()), exploration=1e-5, n_target_obs=100)
    opt = model.optimize()


    # plt.plot(np.cumsum(opt), label='PI Low Exp', color='blue', linestyle='-')
    #
    # # plt.plot(opt1, label='random search')
    # plt.title('Cumulative Squared Errors')
    # plt.legend()
    # # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Squared Error')
    # plt.show()

    print('hello world')
