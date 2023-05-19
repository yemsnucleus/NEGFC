import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from .fake_comp import inject_fake, get_objective_region, create_patch

def log_prob_flux_only(flux):        
    rv_flux = tfp.distributions.Normal(loc=flux, scale=2)
    if tf.math.less(flux, 0.):
        return -np.inf
    return rv_flux.log_prob(flux)

def log_prob_fn(x, y, flux):        
    rv_x = tfp.distributions.Normal(loc=x, scale=.01)
    rv_y = tfp.distributions.Normal(loc=y, scale=.01)
    rv_flux = tfp.distributions.Normal(loc=flux, scale=2)
    return (rv_x.log_prob(x) + rv_y.log_prob(y) + rv_flux.log_prob(flux))

def likelihood_flux_only(flux, x, y, fwhm, cube, patch, rot_ang):    
    fake_cube = inject_fake(x=x, 
                                    y=y, 
                                    flux = -flux, 
                                    cube=cube, 
                                    patch=patch, 
                                    rot_angles=rot_ang)
    obj_reg = get_objective_region(fake_cube, x=x, y=y, rot_ang=rot_ang, fwhm=fwhm)
    return -tf.math.reduce_std(obj_reg)

def likelihood_fn(x, y, flux, fwhm, cube, patch, rot_ang):    
    fake_cube = inject_fake(x=x, 
                                    y=y, 
                                    flux = -flux, 
                                    cube=cube, 
                                    patch=patch, 
                                    rot_angles=rot_ang)
    obj_reg = get_objective_region(fake_cube, x=x, y=y, rot_ang=rot_ang, fwhm=fwhm)
    return -tf.math.reduce_std(obj_reg)

def joint_log_prob_fn(x, y, flux, fwhm, cube, patch, rot_angle):
    
    return log_prob_fn(x, y, flux) + likelihood_fn(x,y,flux, fwhm, cube, patch, rot_angle)

def joint_log_prob_flux_only(flux, x, y, fwhm, cube, patch, rot_angle):
    
    return log_prob_flux_only(flux) + likelihood_flux_only(flux, x,y, fwhm, cube, patch, rot_angle)

@tf.function
def run_chain_only_flux(initial_state, x, y, fwhm, cube, normalized_psf, 
                        rot_ang, num_results=1000, num_burnin_steps=500):
    
    initial_state = [tf.cast(x, tf.float32) for x in initial_state]
    fake_patch = create_patch(cube[0], normalized_psf)
        
    # Define a closure over our joint_log_prob.
    unconstraining_bijectors = [
      tfp.bijectors.Identity()
    ]
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_flux_only(*args, x, y, 
                                                                             fwhm, cube, 
                                                                             fake_patch,
                                                                             rot_ang)
    
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

@tf.function
def run_chain(initial_state, fwhm, cube, normalized_psf, rot_ang, num_results=1000, num_burnin_steps=500):
    initial_state = [tf.cast(x, tf.float32) for x in initial_state]
    fake_patch = create_patch(cube[0], normalized_psf)
        
    # Define a closure over our joint_log_prob.
    unconstraining_bijectors = [
      tfp.bijectors.Identity(),
      tfp.bijectors.Identity(),
      tfp.bijectors.Identity()
    ]
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob_fn(*args, fwhm, cube, fake_patch,rot_ang)
    
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