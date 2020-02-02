import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions
np.random.seed(seed=42)
count_data = np.random.randint(low=0, high=80, size=100)
n_count_data = len(count_data)

def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    alpha = np.array(1./ count_data.mean(), np.float32)
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)
    rv_tau = tfd.uniform()
    lambda_ = tf.gather(
        [lambda_1, lambda_2],
        indices=tf.cast(tau*count_data.size <= np.arange(count_data.size), tf.float32)
    )
    rv_observation = tfd.Poisson(rate=lambda_)
    return (
            rv_lambda_1.log_prob(lambda_1)
            + rv_lambda_2.log_prob(lambda_2)
            + rv_tau.log_prob(tau)
            + tf.reduce_sum(rv_observation.log_prob(count_data))
    )
initial_chain_state = [
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda1"),
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda2"),
    0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
]
unconstraining_bijectors = [
    tfp.bijectors.Exp(),       # Maps a positive real to R.
    tfp.bijectors.Exp(),       # Maps a positive real to R.
    tfp.bijectors.Sigmoid(),   # Maps [0,1] to R.
]
def unnormalized_log_posterior(lambda1, lambda2, tau):
    return joint_log_prob(count_data, lambda1, lambda2, tau)
step_size = tf.Variable(0.05, dtype = tf.float32, name='step_size', trainable=False)
[
    lambda_1_samples,
    lambda_2_samples,
    posterior_tau,
], kernel_results = tfp.mcmc.sample_chain(
    num_results=100000,
    num_burnin_steps=10000,
    current_state=initial_chain_state,
    kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_posterior,
            num_leapfrog_steps=2,
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors))
tau_samples = tf.floor(posterior_tau * tf.cast(tf.size(count_data)), tf.float32)
N = tf.shape(tau_samples)[0]
expected_texts_per_day = tf.zeros(n_count_data)
plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_1$", color=TFColor[0], density=True)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_2$", color=TFColor[6], density=True)