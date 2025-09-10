---
title: Graphical Models - Latent Var. Models
markmap:
  colorFreezeLevel: 4
  initialExpandLevel: 2
---
# Mixture of Gaussians (MoG)
## Generative Model
- **Idea**: Data is generated from a mixture of several Gaussian distributions.
- **Formula**: $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)$
  - $\pi_k$: mixture weight or component prior ($ \sum \pi_k = 1 $)
  - $\mathcal{N}(x_n | \mu_k, \Sigma_k)$: mixture component
  - $K$: number of components
## Maximum Likelihood (ML) Estimation
- **Problem**: Directly maximizing the log-likelihood is difficult.
- **Formula**: $\ln p(X|\pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln \{ \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k) \}$
- **Intuition**: The sum over components $K$ is inside the logarithm, which prevents a simple closed-form solution.
# Expectation-Maximization (EM) Algorithm
## Latent Variable View
- **Idea**: Introduce a hidden (latent) variable $z$ that indicates which component generated a data point.
- **Definition**: $z$ is a K-dimensional binary random variable with a 1-of-K encoding (if point $x_n$ is from component $k$, then $z_{nk}=1$ and other elements are 0).
- **Formulas**:
  - Prior probability of $z$: $p(z_k=1) = \pi_k$
  - Joint distribution: $p(x, z) = p(z)p(x|z) = \prod_{k=1}^K (\pi_k \mathcal{N}(x|\mu_k, \Sigma_k))^{z_k}$
  - Marginalizing gives back the MoG: $p(x) = \sum_z p(x,z)$
- **Responsibility**: The posterior probability of the latent variable.
  - **Formula**: $\gamma(z_k) = p(z_k=1|x) = \frac{\pi_k \mathcal{N}(x|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x|\mu_j, \Sigma_j)}$
  - **Intuition**: $\pi_k$ is the *prior* belief a point comes from component $k$; $\gamma(z_k)$ is the *posterior* belief after observing $x$.
## The General Algorithm
- **Goal**: Maximize likelihood for models with latent variables.
- **Data**:
  - Incomplete data: $X$ (observed)
  - Complete data: $\{X, Z\}$ (observed + latent)
- **Key Idea**: Since we don't know the complete data, we maximize its *expected* log-likelihood.
- **Algorithm**: Iterate between two steps:
  1.  **E-Step (Expectation)**: Compute the posterior of latent variables $p(Z|X, \theta^{old})$ and use it to find the expected complete-data log-likelihood.
      - **Formula (Q-function)**: $Q(\theta, \theta^{old}) = E_{Z|X, \theta^{old}}[\log p(X, Z|\theta)] = \sum_Z p(Z|X, \theta^{old}) \log p(X, Z|\theta)$
  2.  **M-Step (Maximization)**: Find new parameters that maximize the Q-function.
      - **Formula**: $\theta^{new} = \arg\max_\theta Q(\theta, \theta^{old})$
## EM as Lower Bound Maximization
- **Decomposition**: The log-likelihood can be split into two terms.
  - **Formula**: $\log p(X|\theta) = \mathcal{L}(q, \theta) + \text{KL}(q || p)$
    - $\mathcal{L}(q, \theta)$: A lower bound on the log-likelihood (ELBO).
    - $\text{KL}(q || p)$: KL-divergence between an arbitrary distribution $q(Z)$ and the true posterior $p(Z|X, \theta)$.
- **Intuition**: EM is a coordinate ascent algorithm on the lower bound $\mathcal{L}$.
  - **E-Step**: Maximize $\mathcal{L}$ w.r.t $q(Z)$ by setting $q(Z) = p(Z|X, \theta^{old})$. This makes $\text{KL}=0$ and the bound tight.
  - **M-Step**: Maximize $\mathcal{L}$ w.r.t $\theta$, which guarantees that the true log-likelihood also increases.
# EM Algorithm Variations
## MAP-EM
- **Goal**: Find Maximum-A-Posteriori (MAP) solution instead of Maximum Likelihood.
- **Modification**: Add a log-prior term to the M-step.
- **Algorithm**:
  - **E-Step**: Unchanged. Compute $Q(\theta, \theta^{old})$.
  - **M-Step**: $\theta^{new} = \arg\max_\theta [Q(\theta, \theta^{old}) + \log p(\theta)]$
- **Notes**: Useful for regularization. Can prevent ML singularities (e.g., a Gaussian component collapsing on a single point).
## Monte Carlo EM (MC-EM)
- **Problem**: The E-step is intractable; the Q-function integral cannot be computed analytically.
- **Modification**: Approximate the expectation in the E-step with a sample mean.
- **Algorithm**:
  - **E-Step (approximate)**:
    1.  Draw $L$ samples of the latent variables $\{Z^{(l)}\}_{l=1}^L$ from the posterior $p(Z|X, \theta^{old})$.
    2.  Approximate the Q-function: $Q(\theta, \theta^{old}) \approx \frac{1}{L} \sum_{l=1}^L \log p(X, Z^{(l)}|\theta)$.
  - **M-Step**: Maximize the approximate Q-function w.r.t $\theta$.
- **Notes**: Used for more complex models where the E-step lacks a closed-form solution.
# Bayesian Estimation
## Conjugate Priors
- **Problem**: Bayesian inference requires solving integrals, which is often hard.
  - **Formula**: $p(\theta|X) \propto p(X|\theta)p(\theta)$
- **Solution**: Use a **conjugate prior**.
- **Definition**: The prior distribution is conjugate to the likelihood if the posterior has the same functional form as the prior. This makes updates easy.
## Common Distributions & Priors
- **Multinomial - Dirichlet**:
  - **Likelihood (Multinomial)**: Models categorical data with $K$ outcomes.
    - **Formula**: $p(x|\mu) = \prod_{k=1}^K \mu_k^{x_k}$
  - **Conjugate Prior (Dirichlet)**: A distribution over the multinomial parameters $\{\mu_k\}$.
    - **Formula**: $\text{Dir}(\mu|\alpha) \propto \prod_{k=1}^K \mu_k^{\alpha_k - 1}$
- **Gaussian - (Various)**:
  - **Multivariate Gaussian Likelihood**:
    - **Case 1**: Mean $\mu$ unknown, Covariance $\Sigma$ known.
      - **Conjugate Prior**: Gaussian.
    - **Case 2**: Mean $\mu$ known, Precision $\Lambda = \Sigma^{-1}$ unknown.
      - **Conjugate Prior**: Wishart distribution.
    - **Case 3**: Both $\mu$ and $\Lambda$ unknown.
      - **Conjugate Prior**: Gaussian-Wishart distribution.
# Bayesian Mixture Models
## A Full Bayesian Treatment
- **Idea**: Place priors on all parameters of the MoG: weights ($\pi$), means ($\mu_k$), and covariances ($\Sigma_k$).
- **Model Graph**:
  - $\pi \sim \text{Dirichlet}(\alpha)$
  - $(\mu_k, \Sigma_k) \sim \text{Normal-Inverse-Wishart}$
  - $z_n | \pi \sim \text{Multinomial}(\pi)$
  - $x_n | z_n, \mu, \Sigma \sim \mathcal{N}(\mu_{z_n}, \Sigma_{z_n})$
- **Problem**: The posterior over cluster assignments $p(Z|X)$ is intractable.
- **Intuition**: The denominator $\sum_Z p(X|Z)p(Z)$ requires summing over every possible way to partition the data, which is combinatorially explosive.
- **Notes**: This intractability motivates approximate inference methods and leads to models like Dirichlet Processes, which can handle an infinite number of mixture components.
## Dirichlet Concentration Parameter ($\alpha$)
- **Effect**: Controls the properties of the mixture weights $\pi$ drawn from $\text{Dir}(\pi | \alpha)$.
- **Intuition**:
  - Small $\alpha$ (e.g., $\alpha < 1$): Promotes sparsity. Most weights will be near zero, one will be near one.
  - Large $\alpha$ (e.g., $\alpha > 1$): Promotes uniformity. Weights will be of similar value.