---
title: Graphical Models - Approx. Inference
markmap:
  colorFreezeLevel: 4
  initialExpandLevel: 2
---
# Approximate Inference

## Motivation
- **Problem**: Exact Bayesian inference is often intractable.
- **Reasons**:
    - Evaluating the posterior distribution is infeasible.
    - Computing expectations requires integration or summation over exponentially many states.
    - High dimensionality of latent space.
    - Complex form of the posterior distribution.
    - Integrals may not have closed-form solutions.

## Classes of Approximation Schemes
- **Deterministic Approximations (Variational Methods)**
    - **Idea**: Solve a simpler, analytically tractable problem that approximates the true problem.
    - **Notes**:
        - The result is consistently "wrong" but can be computed very quickly.
        - Problem-specific; a new simpler problem must be found for each application.
- **Stochastic Approximations (Sampling Methods)**
    - **Idea**: Approximate the solution using a finite number of samples drawn from the distribution.
    - **Notes**:
        - The approximation becomes more accurate with more samples.
        - It is a generic approach that works for a large number of problems.
        - Computation time scales with the number of samples.

# Markov Random Fields (MRF) for Image Segmentation
## Energy Formulation (MAP Inference)
- **Problem**: Infer the optimal labeling $x$ (true image content) given noisy observations $y$. This is equivalent to finding the Maximum a Posteriori (MAP) solution, which can be achieved by minimizing an energy function.
- **Formula (Energy Function)**:
    - $E(x,y) = \sum_{i} \phi(x_i, y_i) + \sum_{i,j} \psi(x_i, x_j)$
    - The first term is the sum of *unary potentials* (single-node potentials).
    - The second term is the sum of *pairwise potentials*.
- **Unary Potentials $\phi$**:
    - **Intuition**: Encodes local information about a pixel/patch, i.e., how likely it is to belong to a certain class based on its own observation.
    - **Example (Color Model)**: Can be modeled with a Mixture of Gaussians to learn color distributions for each label.
- **Pairwise Potentials $\psi$**:
    - **Intuition**: Encodes neighborhood information, encouraging smoothness by penalizing different labels on adjacent pixels.
    - **Potts Model**:
        - **Formula**: $\psi(x_i, x_j; \theta_\psi) = \theta_\psi \delta(x_i \neq x_j)$
        - **Notes**: A simple model that penalizes any pair of different labels equally.
    - **Contrast-Sensitive Potts Model**:
        - **Formula**:
            - $\psi(x_i, x_j, g_{ij}(y); \theta_\psi) = \theta_\psi g_{ij}(y) \delta(x_i \neq x_j)$
            - where $g_{ij}(y) = e^{-\beta ||y_i - y_j||^2}$
        - **Intuition**: Discourages label changes except in places where the observed pixel values (e.g., colors) also change significantly.

## Solving MRFs with Graph Cuts
- **Goal**: Find the labeling $x$ that minimizes the energy function $E(x)$.
- **Algorithm (Basic Idea)**:
    1. Construct a source-sink (s-t) graph where each pixel is a node.
    2. Design the graph such that any s-t cut corresponds to a binary label assignment for all pixels.
    3. Set the edge weights so that the cost of any cut is equal to the energy of the corresponding labeling.
    4. Find the minimum cost cut using a max-flow/min-cut algorithm. This cut corresponds to the optimal labeling.
- **Condition for Optimality**:
    - **Submodularity**: s-t graph cuts can find the global minimum for binary energy functions only if they are **submodular**.
    - **Formula (Submodularity for pairwise terms)**: $E(s,s) + E(t,t) \leq E(s,t) + E(t,s)$
    - **Intuition**: Submodularity is the discrete equivalent of convexity. It guarantees that any local minimum is also the global minimum.

# Sampling Methods
## Core Idea & Challenges
- **Objective**: Evaluate the expectation of a function $f(z)$ with respect to a probability distribution $p(z)$.
- **Formula (Expectation)**: $E[f] = \int f(z)p(z)dz$
- **Algorithm (Monte Carlo Approximation)**:
    - Draw $L$ independent samples $\{z^{(1)}, ..., z^{(L)}\}$ from $p(z)$.
    - Approximate the expectation with a finite sum: $\hat{f} = \frac{1}{L} \sum_{l=1}^{L} f(z^{(l)})$
- **Notes**:
    - This provides an unbiased estimate of the expectation.
    - **Challenge 1**: Samples may not be truly independent in practice.
    - **Challenge 2**: If $f(z)$ is large where $p(z)$ is small (and vice-versa), the expectation will be dominated by rare events, requiring a very large number of samples for accuracy.

## Sampling from a Distribution (Transformation Method)
- **Problem**: Draw samples from a distribution given its probability density function (pdf) $p(x)$.
- **Algorithm**:
    1.  Compute the cumulative distribution function (CDF): $F(x) = \int_{-\infty}^{x} p(z)dz$.
    2.  Draw a sample $u$ from a uniform distribution $U(0,1)$.
    3.  Transform the sample using the inverse CDF: $x = F^{-1}(u)$. The resulting $x$ is a sample from $p(x)$.
- **Notes**: This method has limited applicability because the inverse CDF $F^{-1}$ is often hard or impossible to compute analytically.

### Efficient Sampling from a Gaussian (Box-Muller Algorithm)
- **Problem**: The CDF of a Gaussian distribution does not have an analytical form, making the transformation method inefficient.
- **Algorithm**: An efficient method to generate pairs of independent, standard normally distributed random numbers from uniformly distributed samples.
- **Multivariate Extension**: To sample from a multivariate Gaussian $N(\mu, \Sigma)$:
    1.  Generate a vector $z$ whose components are independent samples from $N(0,1)$.
    2.  Compute the Cholesky decomposition of the covariance matrix: $\Sigma = LL^T$.
    3.  The final sample is $y = \mu + Lz$.

## Ancestral Sampling
- **Problem**: To draw a sample from a joint distribution represented by a directed graphical model (Bayesian Network).
- **Algorithm**:
    1.  Topologically sort the nodes such that parents come before their children.
    2.  Iterate through the sorted nodes:
        - For each node $x_k$, draw a sample from its conditional probability distribution $p(x_k | \text{pa}_k)$, where $\text{pa}_k$ are the (already sampled) values of its parents.
- **Notes**: Simple and efficient for directed models where sampling from conditionals is easy.

## Rejection Sampling
- **Problem**: To sample from a distribution $p(z)$ when it's difficult to sample from directly, but we can easily evaluate $\tilde{p}(z)$, where $p(z) = \frac{1}{Z_p}\tilde{p}(z)$.
- **Algorithm**:
    1.  Find a simpler proposal distribution $q(z)$ that is easy to sample from.
    2.  Find a constant $k$ such that $kq(z) \geq \tilde{p}(z)$ for all $z$.
    3.  Generate a sample $z_0$ from $q(z)$.
    4.  Generate a random number $u_0$ from the uniform distribution $\text{Uniform}[0, kq(z_0)]$.
    5.  If $u_0 < \tilde{p}(z_0)$, accept $z_0$ as a sample from $p(z)$. Otherwise, reject it and repeat from step 3.
- **Notes**: The efficiency depends on the rejection rate. It becomes extremely inefficient in high-dimensional spaces (curse of dimensionality) because the acceptance probability can become vanishingly small.

## Importance Sampling
- **Problem**: Evaluate the expectation $E[f]$ without being able to draw samples from $p(z)$.
- **Algorithm**:
    1.  Choose a proposal distribution $q(z)$ that is easy to sample from.
    2.  Draw $L$ samples $\{z^{(1)}, ..., z^{(L)}\}$ from $q(z)$.
    3.  Estimate the expectation by a weighted average: $E[f] \approx \sum_{l=1}^{L} w_l f(z^{(l)})$.
- **Formula (Importance Weights)**:
    - If normalization constants are known: $r_l = \frac{p(z^{(l)})}{q(z^{(l)})}$. The estimate is then $\frac{1}{L}\sum r_l f(z^{(l)})$.
    - If normalization constants are unknown ($\tilde{p}, \tilde{q}$):
        - Unnormalized weights: $\tilde{r}_l = \frac{\tilde{p}(z^{(l)})}{\tilde{q}(z^{(l)})}$.
        - Normalized weights: $w_l = \frac{\tilde{r}_l}{\sum_{m=1}^{L} \tilde{r}_m}$.
- **Notes**:
    - All generated samples are used, unlike in rejection sampling.
    - The success depends critically on how well $q(z)$ matches $p(z)$. If $q(z)$ is small or zero where $p(z)$ is significant, the estimate will have high variance or be inaccurate.
    - Also suffers from the curse of dimensionality.

# Markov Chain Monte Carlo (MCMC)
## Core Idea
- **Problem**: Simple sampling methods fail in high dimensions.
- **Idea**: Instead of drawing independent samples, generate a sequence of *dependent* samples $\{z^{(1)}, z^{(2)}, ...\}$ that form a Markov chain.
- **Goal**: Construct the Markov chain so that its stationary (equilibrium) distribution is the target distribution $p(z)$. After a "burn-in" period, the samples from the chain can be used as if they were drawn from $p(z)$.
- **Notes**: Scales much better to high-dimensional problems.

## Metropolis-Hastings Algorithm
- **Idea**: A general framework for constructing an MCMC sampler. It generates a candidate sample from a proposal distribution and then probabilistically accepts or rejects it.
- **Algorithm**:
    1.  Initialize with a starting state $z^{(\tau)}$.
    2.  Generate a candidate state $z^*$ from a proposal distribution $q(z^* | z^{(\tau)})$.
    3.  Calculate the acceptance probability:
        - $A(z^*, z^{(\tau)}) = \min\left(1, \frac{\tilde{p}(z^*)q(z^{(\tau)} | z^*)}{\tilde{p}(z^{(\tau)})q(z^* | z^{(\tau)})}\right)$
    4.  Draw a random number $u \sim \text{Uniform}$.
    5.  If $u < A$, accept the new state: $z^{(\tau+1)} = z^*$.
    6.  Otherwise, reject the new state and keep the old one: $z^{(\tau+1)} = z^{(\tau)}$.
- **Metropolis Algorithm**: A special case where the proposal distribution is symmetric ($q(z_A | z_B) = q(z_B | z_A)$). The acceptance probability simplifies to:
    - $A(z^*, z^{(\tau)}) = \min\left(1, \frac{\tilde{p}(z^*)}{\tilde{p}(z^{(\tau)})}\right)$
- **Notes**:
    - The chain is designed to satisfy detailed balance, which ensures its stationary distribution is $p(z)$.
    - Avoids random walk behavior by accepting "good" moves more often.

## Gibbs Sampling
- **Idea**: A special case of the Metropolis-Hastings algorithm where we sample one variable (or a block of variables) at a time, conditioned on the current values of all other variables.
- **Algorithm**:
    1.  Initialize the state $(z_1, ..., z_D)$.
    2.  For each iteration, cycle through the variables $i=1, ..., D$:
        - Replace the current value of $z_i$ with a new value drawn from its full conditional distribution:
        - $z_i^{(\tau+1)} \sim p(z_i | z_1^{(\tau+1)}, ..., z_{i-1}^{(\tau+1)}, z_{i+1}^{(\tau)}, ..., z_D^{(\tau)})$
- **Notes**:
    - The acceptance probability is always 1.
    - It is parameter-free but requires that we can sample from the full conditional distributions.
    - For graphical models, the conditional for a variable only depends on its Markov blanket, simplifying the process.
    - Can be slow if variables are highly correlated, as it can only make axis-aligned moves.

## Practical Considerations for MCMC
- **Burn-in**: The initial portion of the chain is discarded to allow it to converge to the stationary distribution.
- **Thinning**: To reduce correlation between samples, only every M-th sample is stored and used for estimations.
- **Convergence Diagnostics**: Determining if a chain has run long enough to converge is a major challenge. There are heuristics but no foolproof methods.