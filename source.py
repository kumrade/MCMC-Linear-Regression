import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set standard deviation of error term
sigma_e = 3.0

# Generate synthetic data (same as the previous post, assumed)
np.random.seed(42)
x = np.random.uniform(0, 10, 500)
true_intercept = 0.75
true_slope = 2.0
y = true_intercept + true_slope * x + np.random.normal(0, sigma_e, 500)

# Prior probability: assume Normal priors for both intercept and slope
def prior_log_prob(beta):
    intercept, slope = beta
    log_prior_intercept = norm.logpdf(intercept, loc=0.5, scale=0.5)
    log_prior_slope = norm.logpdf(slope, loc=0.5, scale=0.5)
    return log_prior_intercept + log_prior_slope

# Likelihood: how likely is y given x and beta (a, b)?
def likelihood_log_prob(beta):
    intercept, slope = beta
    y_pred = intercept + slope * x
    log_likelihoods = norm.logpdf(y, loc=y_pred, scale=sigma_e)
    return np.sum(log_likelihoods)

# Posterior probability (log-scale): Prior + Likelihood
def posterior_log_prob(beta):
    return prior_log_prob(beta) + likelihood_log_prob(beta)

# Proposal function: randomly suggest a nearby beta (intercept, slope)
def propose_new_beta(current_beta, step_size=0.5):
    return np.random.normal(loc=current_beta, scale=step_size)

# MCMC sampling using Metropolis-Hastings algorithm
def run_mcmc(start_beta, num_samples=50000, burn_in=10000):
    samples = np.zeros((num_samples, 2))
    samples[0] = start_beta

    for step in range(1, num_samples):
        current_beta = samples[step - 1]
        proposed_beta = propose_new_beta(current_beta)

        # Compute acceptance probability
        log_accept_ratio = posterior_log_prob(proposed_beta) - posterior_log_prob(current_beta)
        accept_prob = np.exp(log_accept_ratio)

        if np.random.rand() < accept_prob:
            samples[step] = proposed_beta  # accept
        else:
            samples[step] = current_beta  # reject (stay)

    return samples[burn_in:]  # discard burn-in samples

# Run the MCMC
initial_guess = [0.5, 0.5]
posterior_samples = run_mcmc(initial_guess)

# Report mean estimates
mean_estimates = posterior_samples.mean(axis=0)
print("Posterior Mean Estimates:")
print(f"Intercept: {mean_estimates[0]:.4f}")
print(f"Slope: {mean_estimates[1]:.4f}")

# Plot histograms of the posterior samples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Intercept
ax1.hist(posterior_samples[:, 0], bins=20, color='skyblue', edgecolor='black')
ax1.axvline(mean_estimates[0], color='red', linestyle='dashed', linewidth=2)
ax1.set_title('Posterior Distribution — Intercept')

# Slope
ax2.hist(posterior_samples[:, 1], bins=20, color='skyblue', edgecolor='black')
ax2.axvline(mean_estimates[1], color='red', linestyle='dashed', linewidth=2)
ax2.set_title('Posterior Distribution — Slope')

plt.tight_layout()
plt.show()
