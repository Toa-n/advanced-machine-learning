---
title: Linear Regression
markmap:
  colorFreezeLevel: 4
  initialExpandLevel: 2
---
# Regression Techniques

## Linear Regression
### Least-Squares Regression
- **Problem**: Given a training set $X = \{x_1, ..., x_N\}$ and target values $T = \{t_1, ..., t_N\}$, learn a function $y(x, w)$ to predict a continuous target value for a new input $x$.
- **Formula**: Minimize the sum-of-squares error function $E(w)$.
    - $E(w) = \frac{1}{2} \sum_{n=1}^{N} \{y(x_n, w) - t_n\}^2$
- **Algorithm**:
    1. Define the error function $E(w)$.
    2. Take the derivative of $E(w)$ with respect to the weights $w_j$.
    3. Set the derivative to zero and solve for $w$.
        - $\frac{\partial E(w)}{\partial w_j} = \sum_{n=1}^{N} \{y(x_n, w) - t_n\} \frac{\partial y(x_n, w)}{\partial w_j} = 0$
- **Notes**:
    - This approach is prone to overfitting, especially on small datasets.

### Overfitting & Regularization
- **Problem**: Standard least-squares can lead to overfitting by fitting the training data too closely, resulting in large coefficient values and poor generalization.
- **Solution**: Add a penalty term (regularizer) to the error function to penalize large coefficient values.

#### Ridge Regression (L2 Regularization)
- **Formula**: The error function is modified with a quadratic (L2) regularizer.
  - $\tilde{E}(w) = \frac{1}{2} \sum_{n=1}^{N} \{y(x_n, w) - t_n\}^2 + \frac{\lambda}{2} ||w||^2$
  - Where $||w||^2 = w^T w = w_0^2 + w_1^2 + ... + w_M^2$
- **Notes**:
  - The parameter $\lambda$ controls the strength of the regularization.
  - The bias term $w_0$ is often omitted from the regularizer.

#### The Lasso (L1 Regularization)
- **Problem**: Solve a regression problem while encouraging sparse coefficients (many coefficients being exactly zero).
- **Formula**: Uses an L1 norm for the regularization term.
  - $w_{Lasso} = \arg\min_w \frac{1}{2} \sum_{n=1}^{N} \{t_n - w^T \phi(x_n)\}^2 + \lambda \sum_{j=1}^{M} |w_j|$
- **Notes**:
  - The L1 penalty makes the problem non-linear, so there is no closed-form solution.
  - It leads to sparse solutions, which is useful for feature selection.
  - The case $q=1$ (Lasso) is the smallest $q$ for which the constraint region is convex.

# A Probabilistic View on Regression
- **Intuition**: View the curve fitting problem from a probabilistic perspective by modeling the uncertainty over the target variable.

## Maximum Likelihood (ML) Estimation
- **Problem**: Given a set of i.i.d. data points $X = \{x_1, ..., x_N\}$, find the parameters $\theta$ of a distribution $p(x|\theta)$ that maximize the likelihood of observing the data.
- **Algorithm**:
    1. Define the likelihood function: $L(\theta) = p(X|\theta) = \prod_{n=1}^{N} p(x_n|\theta)$.
    2. Maximize the log-likelihood by minimizing its negative: $E(\theta) = -\ln L(\theta) = -\sum_{n=1}^{N} \ln p(x_n|\theta)$.
    3. Take the derivative with respect to $\theta$, set to zero, and solve.
- **Notes**:
    - ML can systematically underestimate the variance of the distribution and overfit to the observed data.

### Least-Squares as ML
- **Problem**: Show that minimizing sum-of-squares error is equivalent to maximizing likelihood under a specific assumption.
- **Assumption**: The target variable $t$ is given by a deterministic function $y(x, w)$ plus Gaussian noise $\epsilon$.
  - $t = y(x, w) + \epsilon$, where $\epsilon \sim N(0, \beta^{-1})$
  - This implies $p(t|x, w, \beta) = N(t|y(x, w), \beta^{-1})$
- **Formula**: Maximizing the conditional log-likelihood $\ln p(\mathbf{t}|\mathbf{X}, w, \beta)$ with respect to $w$ is equivalent to minimizing the sum-of-squares error.
  - $w_{ML} = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{t}$
  - The inverse noise precision (variance) is given by: $\frac{1}{\beta_{ML}} = \frac{1}{N} \sum_{n=1}^{N} \{t_n - w_{ML}^T \phi(x_n)\}^2$

## Maximum-A-Posteriori (MAP) Estimation
- **Intuition**: Introduce a prior distribution over the parameters $w$ to control their values and prevent overfitting. Then, find the parameters that maximize the posterior probability.
- **Formula**: Maximize the posterior distribution using Bayes' theorem.
  - $p(w | \mathbf{X}, \mathbf{t}) \propto p(\mathbf{t} | \mathbf{X}, w) p(w)$
- **Algorithm**: Minimize the negative log-posterior.

### Ridge Regression as MAP
- **Assumption**: The prior distribution over the weights $w$ is a zero-mean Gaussian: $p(w|\alpha) = N(w|0, \alpha^{-1}I)$.
- **Formula**: Minimizing the negative log-posterior becomes equivalent to minimizing the regularized sum-of-squares error function:
  - $\frac{\beta}{2} \sum_{n=1}^{N} \{y(x_n, w) - t_n\}^2 + \frac{\alpha}{2} w^T w$
- **Notes**:
  - This is equivalent to Ridge Regression with regularization parameter $\lambda = \alpha / \beta$.

### Lasso as MAP
- **Assumption**: The prior distribution over the weights $w$ is a Laplacian distribution: $p(w) = \frac{1}{2\tau} \exp\{-|w|/\tau\}$.
- **Notes**:
  - A Laplacian prior on the parameters leads to L1 regularization.

## Bayesian Curve Fitting
- **Intuition**: Instead of a single point estimate for parameters $w$, compute a full posterior distribution. Make predictions by averaging over all possible parameter values, weighted by their posterior probability.
- **Problem**: Evaluate the predictive distribution for a new input $x$ by integrating over the posterior of $w$.
- **Formula**:
    - Predictive distribution: $p(t|x, \mathbf{X}, \mathbf{t}) = \int p(t|x, w) p(w|\mathbf{X}, \mathbf{t}) dw$
    - For a Gaussian noise model and Gaussian prior, the predictive distribution is also a Gaussian:
        - $p(t|x, \mathbf{X}, \mathbf{t}) = N(t|m(x), s^2(x))$
    - The mean and variance are:
        - $m(x) = \beta \phi(x)^T \mathbf{S} \sum_{n=1}^{N} \phi(x_n) t_n$
        - $s^2(x) = \beta^{-1} + \phi(x)^T \mathbf{S} \phi(x)$
        - where $\mathbf{S}^{-1} = \alpha \mathbf{I} + \beta \sum_{n=1}^{N} \phi(x_n) \phi(x_n)^T$
- **Notes**:
    - The variance $s^2(x)$ consists of the intrinsic noise in the data ($\beta^{-1}$) and the uncertainty in the parameters $w$.

# Model Evaluation

## Loss Functions
### Minkowski Loss
- **Problem**: Generalize the loss function for regression.
- **Formula**:
    - $L(t, y(x)) = |y(x) - t|^q$
- **Notes**:
    - The optimal prediction that minimizes the expected loss depends on $q$:
        - $q=2$ (Squared loss) -> Conditional Mean $E[t|x]$
        - $q=1$ (Absolute loss) -> Conditional Median
        - $q \to 0$ -> Conditional Mode

## Bias-Variance Decomposition
- **Intuition**: Decompose the expected prediction error to understand different sources of error.
- **Formula**:
    - `expected loss = (bias)^2 + variance + noise`
    - $(\text{bias})^2 = \int \{ E_D[y(x;D)] - h(x) \}^2 p(x) dx$
    - $\text{variance} = \int E_D[\{ y(x;D) - E_D[y(x;D)] \}^2] p(x) dx$
    - $\text{noise} = \iint \{ h(x) - t \}^2 p(x,t) dxdt$
- **Notes**:
    - **Bias**: Error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
    - **Variance**: Error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).
    - There is a trade-off: more complex models tend to have lower bias but higher variance.

# Kernel Methods
- **Intuition**: Implicitly map data into a high-dimensional feature space using a kernel function, then apply a linear model. This allows learning non-linear relationships without explicitly computing the high-dimensional representation.

## Dual Representations
- **Problem**: Reformulate a linear model so it only depends on inner products of feature vectors.
- **Algorithm**:
    1. Express the weight vector as a linear combination of the feature vectors: $w = \Phi^T a$.
    2. Substitute this into the objective function to get a "dual" objective that depends on coefficients $a$.
    3. The dual objective will depend on the Kernel (or Gram) matrix $K = \Phi \Phi^T$.
- **Formula**:
    - Kernel Matrix: $K_{nm} = \phi(x_n)^T \phi(x_m) = k(x_n, x_m)$.

## Kernel Ridge Regression
- **Problem**: Perform non-linear regression using the kernel trick.
- **Formula**:
    - Solution for coefficients `a`:
        - $a = (K + \lambda I_N)^{-1} \mathbf{t}$
    - Prediction for a new input $x$:
        - $y(x) = \mathbf{k}(x)^T (K + \lambda I_N)^{-1} \mathbf{t}$
        - where $\mathbf{k}(x)$ is a vector with elements $k_n(x) = k(x_n, x)$.
- **Notes**:
    - The solution is expressed entirely in terms of the kernel function. This is the **Kernel Trick**.
    - Any algorithm expressible only in terms of inner products can be "kernelized".

## Other Kernel Methods
### Kernel PCA
- **Problem**: Perform Principal Component Analysis in a high-dimensional feature space defined by a kernel.
- **Algorithm**:
    1. This is equivalent to finding the eigenvectors $e'_k$ and eigenvalues $\lambda_k$ of the kernel matrix $K$.
    2. The new coordinate mapping for a sample $x_n$ is $(\sqrt{\lambda_1} e'_{1}, ..., \sqrt{\lambda_K} e'_{K})$.
- **Notes**:
    - Allows for non-linear dimensionality reduction.

# Gaussian Processes (GP)
- **Intuition**: A non-parametric, Bayesian approach that defines a probability distribution directly over functions.

## GP Definition
- **Definition**: A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.
- **Specification**: A GP is fully specified by its mean function $m(x)$ and covariance function (kernel) $k(x, x')$.
    - $m(x) = E[f(x)]$
    - $k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]$
    - Notation: $f(x) \sim GP(m(x), k(x, x'))$

## GP Regression
- **Problem**: Given noisy observations, predict the function value and its uncertainty at new test points.
- **Assumption**: Observations $y$ are the true function values $f$ plus i.i.d. Gaussian noise: $y = f(x) + \epsilon$, where $\epsilon \sim N(0, \sigma_n^2)$.
- **Formula**: The predictive distribution for test points $X_*$ is Gaussian, $p(f_*|X, y, X_*) \sim N(\bar{f_*}, \text{cov}(f_*))$, with:
    - Predictive Mean:
        - $\bar{f_*} = K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} \mathbf{y}$
    - Predictive Covariance:
        - $\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$
- **Notes**:
    - The main computational cost is inverting an $N \times N$ matrix, which is $O(N^3)$.
    - Hyperparameters of the covariance function are typically optimized by maximizing the marginal likelihood.