from matplotlib import pyplot as plt
from gd import gradient_descent

if __name__ == '__main__':

    lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    for a in lr:
        path = gradient_descent(50, a, 0.01)
        plt.plot(path[:,0], path[:,1], label = str(a))
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("gd_random_lr_fun.png")
    plt.show()