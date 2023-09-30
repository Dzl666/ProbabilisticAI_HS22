import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 1
simplefilter("ignore", category=ConvergenceWarning)
""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        self.kernel_f = 0.5* Matern(length_scale=0.5, nu=2.5)+WhiteKernel(0.15)
        self.kernel_v = 1.5 + np.sqrt(2)* Matern(length_scale=0.5, nu=2.5) + WhiteKernel(1e-4)
        
        self.model_f = GPR(self.kernel_f)
        self.model_v = GPR(self.kernel_v)
        # hyperparameter
        self.beta = 3
        self.x_bins = 4000
        # memory
        self.X = []
        self.f = []
        self.v = []

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns - recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns - x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            # randomly generate x0 
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters - x: np.ndarray
            x in domain of f

        Returns - af_value: float
            Value of the acquisition function at x
        """
        mean, std = self.model_f.predict([x], return_std=True)
        speed = self.model_v.predict([x])
        print
        res = mean[0][0] + self.beta*np.sqrt(std[0]) \
            if speed > SAFETY_THRESHOLD else -np.Inf
        return res


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray - Hyperparameters
        f: np.ndarray - Model accuracy
        v: np.ndarray - Model training speed
        """
        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)
        # init
        if not any(self.X):
            self.X = np.empty((0, np.size(x, 1)), float)
            self.f = np.empty((0, np.size(f, 1)), float)
            self.v = np.empty((0, np.size(v, 1)), float)

        self.X = np.vstack((self.X, x))
        self.f = np.vstack((self.f, f))
        self.v = np.vstack((self.v, v))
        self.model_f.fit(self.X, self.f)
        self.model_v.fit(self.X, self.v)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns - solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        # partition
        x_domain = np.linspace(*domain[0], self.x_bins)[:, None]
        # init
        f_max = -np.Inf
        x_opt = 0
        # check each x for f and v
        for _, x_t in enumerate(x_domain):
            mean, std = self.model_f.predict([x_t], return_std=True)
            speed = self.model_v.predict([x_t])
            f_t = mean + self.beta* np.sqrt(std)
            if f_t > f_max and speed > SAFETY_THRESHOLD:
                f_max = f_t
                x_opt = x_t
        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2
def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()
    np.random.seed(SEED)

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')

if __name__ == "__main__":
    main()