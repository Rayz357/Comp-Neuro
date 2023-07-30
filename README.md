# Comp-Neuro
My learning progress in Computational Neuroscience

# 7-26:
- Neuromatch: Intro to models - What models
- what to 
# 7-28:
- "How" Models
  - How does a neuron spike? How does the Number of spikes-Inter spike intervals came out?
  - linear integrate-and-fire model neuron
  - Spiking Inputs: I∼Poisson(λ)
  - Inter Spike Intervals: alpha, rate -> number of intervals
 
  - Inhibitory signals:
  - update model: dVm=−βVm+αI   (Leaky Integrate-and-Fire model neuron)  
I=Iexc−Iinh  
Iexc∼Poisson(λexc)  
Iinh∼Poisson(λinh)  
dv = alpha*(exc[i]-inh[i])-beta*v[i-1]  
summary: In this tutorial we gained some intuition for the mechanisms that produce the observed behavior in our real neural data. First, we built a simple neuron model with excitatory input and saw that its behavior, measured using the ISI distribution, did not match our real neurons. We then improved our model by adding leakiness and inhibitory input. The behavior of this balanced model was much closer to the real neural data.  

Cmd\*dVm/dt=−(Vm−Vrest)/Rm+I  
- Why model: an optimization problem - to get the biggest information content  
- if our neuron has a fixed budget, what ISI distribution should it express (all else being equal) to maximize the information content of its outputs?  
# 7-29: W1D2 fitting models to data
  Tutorial 1: Linear regression with MSE  
  minimize  MSE by solving the normal equations using calculus, we find that θ^=x⊤y/x⊤x.  
  (set the derivative of the error expression with respect to  θ  equal to zero)
  Python skills:  
  - np.means: get mean of matrix calculations(axis=0, 1 , ...)  
  - ndarray tranpose: x.T  
  - ndarray 相乘: x@y  

# 7-30: W1D2 
## Tutorial 2:  Linear regression with MLE  
- Learn about probability distributions and probabilistic models
- Learn how to calculate the likelihood of our model parameters
- Learn how to implement the maximum likelihood estimator, to find the model parameter with the maximum likelihood  
Likelihood vs probability:
L(θ|x,y)=p(y|x,θ)  
p(y|x,θ)→  "probability of observing the response  y  given parameter  θ  and input  x "  
L(θ|x,y)→  "likelihood model that parameters  θ  produced response  y  from input  x "
  
## Tutorial 3:Confidence intervals and bootstrapping  
In this tutorial, we will discuss how to gauge how good our estimated model parameters are.  
- Learn how to use bootstrapping to generate new sample datasets
- Estimate our model parameter on these new sample datasets
- Quantify the variance of our estimate using confidence intervals
### Boostrapping
- Bootstrapping is a resampling procedure that allows to build confidence intervals around inferred parameter values. The idea is to generate many new synthetic datasets from the initial true dataset by randomly sampling from it, then finding estimators for each one of these new datasets, and finally looking at the distribution of all these estimators to quantify our confidence.
- use ```numpy.random.choice``` method to resample a dataset with replacement. The actual number of points is the same, but some have been repeated so they only display once.
- Generate a set of theta_hat estimates using the bootstrap method, invoke the function solve_normal_eqn to produce the MSE-based estimator.
- Quantify how uncertain our estimated slope is by computing confidence intervals (CIs) from our bootstrapped estimates. The most direct approach is to compute percentiles from the empirical distribution of bootstrapped estimates.

