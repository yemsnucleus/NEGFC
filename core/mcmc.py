import tensorflow_probability as tfp
import tensorflow as tf 
import pandas  as pd
import os


def log_prob_fn(flux):        
	# rv_x = tfp.distributions.Normal(loc=x, scale=.1)
	# rv_y = tfp.distributions.Normal(loc=y, scale=.1)
	rv_flux = tfp.distributions.Normal(loc=flux, scale=2)
	# return (rv_x.log_prob(x) + rv_y.log_prob(y) + rv_flux.log_prob(flux))
	return rv_flux.log_prob(flux)

def likelihood_fn(flux, cube, psf, rot_ang, back_med, back_std):   
	scaled_psf = tf.cast(psf, tf.float32) * flux
	
	residuals = tf.expand_dims(cube, -1) - scaled_psf

	residuals_std = tf.math.reduce_std(residuals)
	# residuals_med = tfp.stats.percentile(residuals, q=50)

	# loss_std = tf.pow(back_std - residuals_std, 2)
	# loss_med = tf.pow(back_med - residuals_med, 2)
	return residuals_std

def joint_log_prob_fn(flux, cube, psf, rot_angle, back_med, back_std):
	return log_prob_fn(flux) + likelihood_fn(flux, cube, psf, rot_angle, back_med, back_std)


@tf.function
def run_chain(initial_flux, cube, psf_ext, rot_ang, num_results=1000, num_burnin_steps=500):
	initial_state = [tf.cast(initial_flux, tf.float32)]

	# Background stats
	max_value = tfp.stats.percentile(cube, q=25)
	mask = tf.where(cube < max_value, 1., 0.)
	p25_companion = cube * mask
	values = tf.reshape(p25_companion, [-1])
	print(values.shape)
	non_zero_mask = tf.where(values != 0, True, False)
	valid = tf.boolean_mask(values, non_zero_mask)

	back_med = tfp.stats.percentile(valid, q=50)
	back_std = tf.math.reduce_std(valid)

	# Define a closure over our joint_log_prob.
	unconstraining_bijectors = [
	  # tfp.bijectors.Identity(), # x pos
	  # tfp.bijectors.Identity(),	# y pos
	  tfp.bijectors.Identity(), # flux
	]

	unnormalized_posterior_log_prob = lambda *args: joint_log_prob_fn(*args, cube, psf_ext, rot_ang, back_med, back_std)
	
	# Define the HMC transition kernel.
	hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
	  target_log_prob_fn=unnormalized_posterior_log_prob,
	  num_leapfrog_steps=2,
	  step_size=0.1)
	
	# Define the MCMC transition kernel.
	mcmc_kernel = tfp.mcmc.TransformedTransitionKernel(
	  inner_kernel=hmc_kernel,
	  bijector=unconstraining_bijectors)
	
	# Define the initial state of the chain.
	current_state = initial_state
	
	# Sample from the chain.
	samples = tfp.mcmc.sample_chain(
				  num_results=num_results,
				  num_burnin_steps=num_burnin_steps,
				  current_state=current_state,
				  kernel=mcmc_kernel)

	return samples
