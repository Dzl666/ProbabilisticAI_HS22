import os
from pickle import TRUE
import typing
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from matplotlib import cm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """
    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        # TODO: Add custom initialization for your model here if necessary
        # select and compose kernel
        self.kernel = ConstantKernel(constant_value=0.3, constant_value_bounds=(1e-3, 1)) \
                     * RBF(length_scale=0.05, length_scale_bounds='fixed')
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            n_restarts_optimizer=2
        )
        self.kernelApprxi_model = SGDClassifier()
        self.feature_map = RBFSampler(
            gamma=0.1,
            n_components=200,
            random_state=1
        )
        # self.sigma_p = 0.28
        # self.length_scale = 0.06
        # self.sigma_n = 1e-10
    
    def gaussian_kernel(self, xi, xj):
        return self.sigma_p**2 * np.exp(-0.5 *cdist(xi,xj)**2 /self.length_scale**2)

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # Use GP to estimate the posterior mean and stddev for each location
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)

        # K_AA = self.gaussian_kernel(self.train_X, self.train_X)
        # k_xx = self.gaussian_kernel(test_features, test_features)
        # k_xA = self.gaussian_kernel(self.train_X, test_features)
        # K_AA_inv = np.linalg.inv(K_AA + self.sigma_n *np.eye(self.train_X.shape[0]))
        # gp_mean = (k_xA.T).dot(K_AA_inv).dot(self.train_Y) * 1.04
        # gp_std = k_xx - (k_xA.T).dot(K_AA_inv).dot(k_xA)

        gp_mean, gp_std = self.model.predict(test_features, return_std=True)
        #self.kernelApprxi_model.predict(test_features, return_std=True)

        # Use the GP posterior to form predictions
        predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        choice_p = 0.6 #self.rng.random()
        train_set = np.random.choice([True, False], len(train_GT), p=[choice_p, 1.0-choice_p])
        self.train_X = train_features[train_set,:]
        print('Training samples size: ' + str(self.train_X.shape[0]))
        self.train_Y = train_GT[train_set]
        self.model.fit(self.train_X, self.train_Y)

        #X_features = self.feature_map.fit_transform(train_X)
        #print(X_features.shape)
        #self.kernelApprxi_model.fit(X_features, train_Y)


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL
    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT
    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)

def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


# BASELINE COST: 118.556
def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
        # train_y.csv contain additional measurement noise
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)
        # the test data does not contain measurement noise

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT, train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')

if __name__ == "__main__":
    main()
