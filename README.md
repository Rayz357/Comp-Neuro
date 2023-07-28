# Comp-Neuro
My learning progress in Computational Neuroscience

7-26:
- Neuromatch: Intro to models - What models
- what to 
7-28:
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
Why model: an optimization problem - to get the biggest information content  
if our neuron has a fixed budget, what ISI distribution should it express (all else being equal) to maximize the information content of its outputs?  
