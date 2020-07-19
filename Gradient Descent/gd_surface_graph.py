
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X1 = np.arange(-5, 5, 0.25)
    X2 = np.arange(-5, 5, 0.25)
    X1, X2 = np.meshgrid(X1, X2)
    Z = pow(X1, 2) + pow(X2, 2) + X1 * X2

    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf.show()