"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, DotProduct
from scipy.stats import norm
import matplotlib.pyplot as plt
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA --> adjust to 0, to make gaussian process prior correct


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
# preprocess data by subtracting 4 
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        self.var_f = 0.0225
        self.var_v = 1e-8

        self.kernel_f = 1.0*Matern(nu=2.5) + WhiteKernel(noise_level=self.var_f)
        self.kernel_v = DotProduct() + 1.0*Matern(nu=2.5) + WhiteKernel(self.var_v) #+ 4 #technically doesn't account for the prior mean = 4 but hey
        
         # --> subtract 4 off all v(x) values to normalize

        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v)

        self.X = np.array([]) 
        self.F = np.array([]) 
        self.V = np.array([])

        self.max_x = None
        self.max_f = None # f value belonging to x
        self.max_v = None # v val belonging to x, only for double checking purposes

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # multiple different starting points x0
        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    # written to be maximized
    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        f_mean, f_std = self.gp_f.predict(x, return_std=True)
        v_mean, v_std = self.gp_v.predict(x, return_std=True)

        #COMPUTE EI according to paper
        # flip Z so that we maximize instead of minimize (max_f - mean instead of mean-f_max)

        Z = (f_mean-self.max_f)/(f_std) # unsure if Z or -Z

        EI = f_std*(Z*norm.cdf(Z) + norm.pdf(Z))

        # Probability of not violating bound (P(v(x) < 4))
        # subtract 2*v_std to make it harder to accept points --> unsafe evals are punished harshly in the score
        PF = norm.cdf(0-2.0*v_std, loc=v_mean, scale=v_std) # norm v, 0 instead of SAFETY THRESHOLD
        # PF = norm.cdf(SAFETY_THRESHOLD-2.0*v_std, loc=v_mean, scale=v_std)

        combined_acquisition = (
            # self.lambda_ * ei + (1 - self.lambda_) * pi - self.beta * v_violation**2
            EI*PF # square PF to further punish probability of violating the safety threshold
        )

        return combined_acquisition

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # norm v by subtracting safety therhold:
        
        v -= SAFETY_THRESHOLD

        # print(f"normalized v: {v}")

        self.X = np.append(self.X, x)
        self.F = np.append(self.F, f)
        self.V = np.append(self.V, v) 

        self.gp_f.fit(X=self.X.reshape(-1, 1), y=self.F.reshape(-1, 1))
        self.gp_v.fit(X=self.X.reshape(-1, 1), y=self.V.reshape(-1, 1))

        print(self.X.shape)

        # prior 4 needs to be accounted for somehow
        if (v < 0): # norm v, subtract safety threshold
        # if (v < SAFETY_THRESHOLD):
            if ((self.max_x is None) or (self.max_f < f)):
                self.max_x = x
                self.max_f = f
                self.max_v = v

        self.plot()

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        return self.max_x

# doesn't work :(
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """

        X_test = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 500).reshape(-1, 1)
        f_mean, f_std = self.gp_f.predict(X_test, return_std=True)
        v_mean, v_std = self.gp_v.predict(X_test, return_std=True)
        plt.figure(figsize=(12, 6))

        # Plot GP for f(x)
        plt.subplot(1, 2, 1)
        plt.title("GP for f(x)")
        plt.fill_between(
            X_test.ravel(), 
            f_mean - 1.96 * f_std, 
            f_mean + 1.96 * f_std, 
            color="lightblue", alpha=0.5, label="95% CI"
        )
        plt.plot(X_test, f_mean, "b-", label="Mean Prediction")
        plt.scatter(self.X, self.F, c="red", label="Observations", zorder=5)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()

        # Plot GP for v(x)
        plt.subplot(1, 2, 2)
        plt.title("GP for v(x)")
        plt.fill_between(
            X_test.ravel(), 
            v_mean - 1.96 * v_std, 
            v_mean + 1.96 * v_std, 
            color="lightgreen", alpha=0.5, label="95% CI"
        )
        plt.plot(X_test, v_mean, "g-", label="Mean Prediction")
        plt.scatter(self.X, self.V, c="orange", label="Observations", zorder=5)
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.legend()

        plt.tight_layout()
        plt.show()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0

# not used during actual eval
def get_initial_safe_point():
    """Return initial safe point"""
    print("__________________________________________") # make terminal output prettier 
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # unsafe evals
    unsafe_evals = 0

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        # print("___________________________________________")
        # print("Iteration", j)
        # print("MAX X:", agent.max_x, "MAX VAL F:", agent.max_f, "SA:", agent.max_v)
        x = agent.recommend_next()

        # Check for valid shape
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function recommend_next must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        if (cost_val >= 4):
            unsafe_evals += 1
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals {unsafe_evals}\n')


if __name__ == "__main__":
    main()
