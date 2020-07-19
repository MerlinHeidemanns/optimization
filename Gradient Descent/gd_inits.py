import sys
from matplotlib import pyplot as plt
from gd import gradient_descent

if __name__ == '__main__':

    iter = int(sys.argv[1])
    alpha = float(sys.argv[2])
    epsilon = float(sys.argv[3])

    fig, ax = plt.subplots()
    for i in range(10):
        path = gradient_descent(iter, alpha, epsilon)
        ax.plot(path[:,0], path[:,1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.savefig("gd_random_init_fun.png")
    plt.show()
