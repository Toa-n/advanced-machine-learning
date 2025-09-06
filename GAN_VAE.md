---
title: Generative Models 1
markmap:
  colorFreezeLevel: 4
  initialExpandLevel: 2
---
# Generative Adversarial Networks (GANs)
## Concept
- **Idea**: Two networks compete to generate realistic data
- **Analogy**: Two-player game
## Components
- ### Generator (G)
	- **Goal**: Create data from random noise to fool the Discriminator
	- **Analogy**: Counterfeiter
- ### Discriminator (D)
	- **Goal**: Distinguish between real data and generated (fake) data
	- **Analogy**: Police investigator
- **Note**: Both are deep networks trained with backpropagation
## Training Procedure
- ### Train Discriminator
	1. Freeze Generator weights
	2. Feed real and fake (generated) images to Discriminator
	3. Update Discriminator weights to improve classification (real vs fake)
- ### Train Generator
	1. Freeze Discriminator weights
	2. Generate fake images
	3. Update Generator weights to make its output more "real" according to the Discriminator
## Formalization
- ### Minimax Game
	- **Formula**: $\min_G \max_D V(D, G) = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_z(z)}[\log(1 - D(G(z)))]$
	- **Intuition**: G minimizes the objective while D maximizes it
- ### Nash Equilibrium
	- **Condition**: Achieved when the generated distribution equals the real data distribution
	- **Formulas**:
		- $p_g(x) = p_{data}(x)$
		- $D(x) = 1/2$ (Discriminator cannot distinguish)
- ### Algorithm: Minibatch SGD
	- #### Discriminator Update (k steps)
		- **Gradient Ascent**: $\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^{m} [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$
	- #### Generator Update (1 step)
		- **Gradient Descent**: $\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^{m} \log(1 - D(G(z^{(i)})))$
## Applications & Extensions
- ### Deep Convolutional GAN (DCGAN)
	- **Architecture**:
		- No fully-connected layers
		- Uses strided convolutions (discriminator) and fractional-strided convolutions (generator)
		- Uses Batch Normalization
		- ReLU and Leaky ReLU activations
- ### Conditional GAN (cGAN)
	- **Idea**: Condition the generator and discriminator on extra information (e.g., an input image or class label)
	- **Applications**: Image-to-image translation (pix2pix), text-to-image
- ### Super-Resolution (SRGAN)
	- **Application**: Upscaling low-resolution images to high-resolution
## Problems
- ### Vanishing Gradients
	- **Problem**: If the discriminator is too good, its loss gradient drops to zero
	- **Result**: The generator gets no feedback for learning
- ### Mode Collapse
	- **Problem**: The generator produces a limited variety of samples, failing to capture the full data distribution
- ### Non-convergence
	- **Problem**: The two-player training dynamic is unstable and may not converge to the Nash equilibrium
- ### Low-dimensional Support
	- **Problem**: The distributions of real and generated data might exist on manifolds that do not intersect, making the default loss function ineffective
	- **Solution**: Wasserstein GANs (WGANs) use a different loss function (Earth Mover's distance) to handle this

# Variational Autoencoders (VAEs)
## Autoencoders (AE)
- **Concept**: An unsupervised neural network for dimensionality reduction
- **Components**:
	- **Encoder**: Compresses input $x$ into a latent representation $z$
	- **Decoder**: Reconstructs the input $\hat{x}$ from $z$
- **Training**:
	- **Goal**: Minimize reconstruction error
	- **Loss Function**: $L_2$ loss, $||x - \hat{x}||^2$
- **Variants**:
	- ### Regularized AE
		- **Problem**: Powerful AEs can just copy the input without learning useful features
		- **Solution**: Add a regularization term to the loss on the latent space $z$
		- **Formula**: $L(x, g(f(x))) + \Omega(z)$
	- ### Denoising AE (DAE)
		- **Idea**: Train the AE to reconstruct the original input $x$ from a corrupted version $\tilde{x}$
		- **Effect**: Forces the model to learn the underlying data manifold
## Probabilistic View
- **Goal**: Build a generative model that can sample new data
- **Process**:
	1. Sample latent variable $z$ from a prior distribution $p(z)$
	2. Generate data $x$ from a conditional distribution $p_\theta(x|z)$ (the decoder)
- **Problem**: The data likelihood is intractable to compute
	- **Formula**: $p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$
## VAE Solution
- **Idea**: Introduce an encoder $q_\phi(z|x)$ to approximate the true (but intractable) posterior $p_\theta(z|x)$
- **Evidence Lower Bound (ELBO)**:
	- **Concept**: Instead of maximizing the true likelihood, we maximize a tractable lower bound on it
	- **Formula**: $\log p_\theta(x) \ge E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z))$
	- **Interpretation**:
		- **Reconstruction Term**: $E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$ makes the output look like the input
		- **Regularization Term**: $D_{KL}(q_\phi(z|x) || p_\theta(z))$ forces the learned latent distribution to be close to the simple prior (e.g., a standard Gaussian)
## Latent Space
- **AE**: Can be messy, with points memorized in disconnected regions
- **VAE**: The regularization term forces a smooth, continuous latent space, which is better for generation
## Extensions
- ### VAE-GAN
	- **Idea**: Combine the sharp samples from GANs with the stable training of VAEs
	- **Method**: Use the feature-matching loss from a GAN discriminator as the reconstruction loss for a VAE
- ### Vector Quantized VAE (VQ-VAE)
	- **Problem**: VAEs learn a continuous latent space, but many data types are inherently discrete
	- **Solution**: Use a discrete codebook of embedding vectors to quantize the encoder's output
	- **Training Loss**:
		- **Formula**: $\log(p(x|z_q(x))) + ||\text{sg}[z_e(x)] - e||^2 + \beta ||z_e(x) - \text{sg}[e]||^2$
- ### VQ-VAE-2 & DALL-E
	- **VQ-VAE-2**: A hierarchical version of VQ-VAE for high-fidelity image generation
	- **DALL-E**: A large transformer model trained on text-image pairs that uses VQ-VAE principles to generate images from text descriptions
- ### Hierarchical VAEs
	- **Problem**: "Flat" VAEs are limited by simple priors
	- **Solution**: Use a hierarchy of latent variables, creating a more powerful generative model

# Denoising Diffusion Models
## Concept
- **Idea**: A generative model based on two processes
- ### Forward Diffusion Process
	- **Goal**: Systematically and slowly destroy structure in data by adding Gaussian noise over $T$ steps
	- **Process**: Fixed, not learned
- ### Reverse Denoising Process
	- **Goal**: Learn to reverse the diffusion process, turning noise into data
	- **Process**: Learned with a neural network
## Forward Process (Noising)
- **Single Step**:
	- **Formula**: $q(x_t|x_{t-1}) = N(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$
	- **Note**: $\beta_t$ is from a predefined "noise schedule"
- **Arbitrary Step (Diffusion Kernel)**:
	- **Concept**: We can sample $x_t$ at any step $t$ directly from the original data $x_0$
	- **Formula**: $q(x_t|x_0) = N(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$
		- where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$
	- **Reparametrization Trick**:
		- **Formula**: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$, where $\epsilon \sim N(0, I)$
- **Final State**: After T steps, $x_T$ is approximately pure Gaussian noise, $x_T \approx N(0, I)$
## Reverse Process (Denoising)
- **Goal**: Learn the intractable true reverse distribution $q(x_{t-1}|x_t)$
- **Solution**: Approximate it with a neural network $p_\theta(x_{t-1}|x_t)$
- **Model**:
	- **Formula**: $p_\theta(x_{t-1}|x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$
	- **Architecture**: A network (typically a U-Net) is trained to predict the noise $\epsilon$ that was added at step $t$. From this prediction, the mean $\mu_\theta$ can be calculated.
- **Generation**:
	1. Start with a sample from a standard Gaussian distribution, $x_T \sim N(0, I)$
	2. Iteratively use the trained network to denoise for $T$ steps, from $t=T$ down to $t=1$, to get a clean data sample $x_0$