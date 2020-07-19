# Gradient Descent

Gradient descent is an optimization algorithm. Given a differentiable function $f:\mathbb{R}^d \rightarrow \mathbb{R}$ the vector of partial derivatives is given by $\nabla f(\mathbf{w}) = \bigg(\frac{\partial f(\mathbf{w})}{\partial w_1}\dots\frac{\partial f(\mathbf{w})}{\partial w_d} \bigg)$. The gradient points in the direction of the largest increase of the function given some vector $\mathbf{w}$ such that to minimize an objective function, the gradient is subtracted from the previous weight vector.
$$
\mathbf{w}^t = \mathbf{w}^{t - 1} - \nabla f(\mathbf{w}^{t - 1})
$$
Generally, we adjust the algorithm with a learning rate $\alpha$ to adjust the size of each step towards a minimum such that
$$
\mathbf{w}^t = \mathbf{w}^{t - 1} - \alpha\nabla f(\mathbf{w}^{t - 1})
$$

## Examples

### Convex Function

Below is a Python implementation. We take a simple function $f(x_1, x_2) = x_1^2 + x_2^2 + x_1x_2$ with gradient $\nabla f(\mathbf{x}) = \begin{pmatrix}2x_1 + x_2 & 2 x_2 + x_1\end{pmatrix}$.

```python
def gradient_descent(iter, alpha, epsilon):
    x = np.random.normal(loc = 0, scale = 3, size=(1, 2))
    i = 0
    flag = True
    while i < iter and flag:
        i += 1
        gradient = np.array([[
            2 * x[0, 0] + x[0, 1],
            2 * x[0, 1] + x[0,0]
        ]])
        x = x - alpha * gradient
        if abs(np.sum(gradient, axis = 1)) < epsilon:
            flag = False
    return x
```

The function takes three arguments. `iter` , the number of iterations, `alpha` , the learning rate, and `epsilon`, the convergence criterion. The function terminates after either A) a set number of iterations or B) the absolute value of sum of the gradients is smaller than the convergence criterion. 

In practice, we randomly draw starting values for `x` and initialize the counter `i` and the flag for the convergence criterion. At each step, we calculate the gradient given the *previous* values of `x` (rather than to update each value of `x` one at a time and then calculate the gradient for the other with the partially updated `x` vector). Next, we subtract the gradient scaled by `alpha` from `x`. Lastly, we check for the convergence criterion.

With ten random initializations, this looks like this.

![gd_random_init_fun](/Users/merlinheidemanns/Documents/Research/study_self/methods/optimization/optimization_algos/gd_random_init_fun.png)

#### Learning rate

The learning rate determines the size of the step, the algorithm takes in the opposite direction of the gradient. Fixing the number of iterations and allowing for random initializations while varying the learning rate, the figure below shows variation in the behavior of the algorithm. For very small learning rates, the minimum remains unreached. For large step sizes, the path becomes jagged.

![gd_random_lr_fun](/Users/merlinheidemanns/Documents/Research/study_self/methods/optimization/optimization_algos/gd_random_lr_fun.png)

