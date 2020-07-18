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

In Python, we can write a gradient descent algorithm as follow

