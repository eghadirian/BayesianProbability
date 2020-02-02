import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
rv_p = tfd.Uniform(low=0., high=1., name='p')
prob_true = 0.05
N = 1500
occurrences = tfd.Bernoulli(probs=prob_true).sample(sample_shape=N, seed=10)
occurrences_sum = tf.reduce_sum(occurrences)
occurrences_mean = tf.reduce_mean(tf.cast(occurrences,tf.float32))


def joint_log_prob(occurrences, prob_A):
    rv_prob_A = tfd.Uniform(low=0., high=1.)
    rv_occurrences = tfd.Bernoulli(probs=prob_A)
    return (
            rv_prob_A.log_prob(prob_A)
            + tf.reduce_sum(rv_occurrences.log_prob(occurrences))
    )
number_of_steps = 48000
burnin = 25000
leapfrog_steps=2
initial_chain_state = [
    tf.reduce_mean(tf.cast(occurrences, dtype=tf.float32))
    * tf.ones([], dtype=tf.float32, name="init_prob_A")
]
unconstraining_bijectors = [
    tfp.bijectors.Identity()
]
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(occurrences, *args)
step_size = tf.Variable(0.5, trainable=False, dtype=tf.float32, name='step_size')
hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=leapfrog_steps,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=int(burnin * 0.8)),
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)
[
    posterior_prob_A
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)
burned_prob_A_trace_ = posterior_prob_A[burnin:]
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(prob_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(burned_prob_A_trace_, bins=25, histtype="stepfilled", normed=True)
plt.show()