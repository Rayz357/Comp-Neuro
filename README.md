# Comp-Neuro
My learning progress in Computational Neuroscience

# 
- Neuromatch: Intro to models - What models
- what to 
# 
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
#  W1D2 fitting models to data
  Tutorial 1: Linear regression with MSE  
  minimize  MSE by solving the normal equations using calculus, we find that θ^=x⊤y/x⊤x.  
  (set the derivative of the error expression with respect to  θ  equal to zero)
  Python skills:  
  - np.means: get mean of matrix calculations(axis=0, 1 , ...)  
  - ndarray tranpose: x.T  
  - ndarray 相乘: x@y  

#  W1D2 
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
## Tutorial 4: Multiple linear regression and polynomial regression
we will generalize the regression model to incorporate multiple features.  
- Learn how to structure inputs for regression using the 'Design Matrix'
- Generalize the MSE for multiple features using the ordinary least squares estimator
- Visualize data and model fit in multiple dimensions
- Fit polynomial regression models of different complexity
- Plot and evaluate the polynomial regression fits

# 
## Tutorial 5:bias-variance trade-off
- Bias-Variance Trade-off:   
We need to strike the right balance between bias and variance. Ideally we want to find a model with optimal model complexity that has both low bias and low variance  
Too complex models have low bias and high variance.  
Too simple models have high bias and low variance.  
##  Model Selection: Cross-validation
We need to use model selection methods to determine the best model to use for a given problem.  
Cross-validation focuses on how well the model predicts new data.    
 ### k-fold cross-validation
- use Validation data to select the best model during the training process so that we don't have to use the test data

## Tutorial outro
- Cross validation VS. Bootstrapping
- in bootstrapping, we generate many models(parameters) and see how the estimates distributes. By computing confidence intervals (CIs) from our bootstrapped estimates, we can?
- in Cross validation, we select models by asking how well the model predicts new data. and k-fold cross validation allows us to use train data as validation data.

# Generalized linear models
## Prequiresites:
- Stimulus Representation:
- Stochatistic Inference
# 
## Tutorial 1: GLMs for Encoding
OBJ: model a retinal ganglion cell spike train by fitting a temporal receptive field, first with Linear-Gaussian GLM (ordinary least-squares regression model), then with Poisson GLM  

#  Tutorial 1: GLMs for Encoding
- make a zero-padded version of the stimulus
- initialize an empty design matrix with the correct shape
- fill in each row of the design matrix, using the zero-padded version of the stimulus
# Tutorial 1: 
## Section 1.1: Load retinal ganglion cell activity data
在这个练习中，我们使用来自实验的数据。该实验使用一个屏幕，该屏幕在两个亮度值之间随机交替，并记录了视网膜神经节细胞（RGC）的反应，RGC是眼睛后部视网膜中的一种神经元。这种视觉刺激被称为“全场闪烁”(“full-field flicker)，它以120Hz的频率呈现(~8ms)。该time bin用于计算每个神经元发出的spikes数量。  
RGCdata.mat文档包含三个变量：
- 刺激Stim :每个时间点的刺激强度。它是一个形状为 T×1 的数组(T=144051) 
- SpCounts，2 个 ON 细胞和 2 个 OFF 细胞的在每个time bin的发出的尖峰数（binned spike counts)。它是一个 144051×4 数组，每列代表不同细胞的spike counts
- dtStim，单个time bin的大小（以秒为单位），以峰值/秒为单位计算模型输出。尖峰帧速率(stimulus frame rate) = 1 / dtStim 
```python
data = loadmat('RGCdata.mat')  # loadmat is a function in scipy.io
dt_stim = data['dtStim'].item()  # .item extracts a scalar value

# Extract the stimulus intensity
stim = data['Stim'].squeeze()  # .squeeze removes dimensions with 1 element 移除长度为1的维度，如(N,1)->(N,)

# Extract the spike counts for one cell
cellnum = 2
spikes = data['SpCounts'][:, cellnum]

# Don't use all of the timepoints in the dataset, for speed
keep_timepoints = 20000
stim = stim[:keep_timepoints]
spikes = spikes[:keep_timepoints]
#Use the plot_stim_and_spikes helper function to visualize the changes in stimulus intensities and spike counts over time.
plot_stim_and_spikes(stim, spikes, dt_stim)
```
### Coding Exercise 1.1: Create design matrix
我们的目标是根据细胞之前的刺激强度来预测细胞的活动。这将帮助我们了解 RGC 如何随着时间的推移处理信息。为此，我们首先需要为此模型创建设计矩阵，该矩阵以矩阵形式组织刺激强度，使得第 i 行具有时间点 i 之前的刺激帧（细胞接受的刺激数量）。在本练习中，我们将使用 d=25 时间滞后(time bin?)创建设计矩阵 X。也就是说，X应该是T×d矩阵。 X是有T个细胞在d=25个时间滞后中产生的刺激d=25（约 200 ms）是我们根据影响 RGC 响应的时间窗口的先验知识做出的选择。  
行 t 中的最后一个元素应对应于在时间 t 产生的刺激，其左侧的元素应包含前一次time bin输入的stim值，等等。  


具体来说，Xij 将是时间 i+d-1-j 时的刺激强度 。请注意，在前几个时间段中，我们可以访问记录的尖峰计数，但不能访问输入的刺激值。为简单起见，我们假设数据集中第一个时间点之前的时间滞后的 stim 值为 0。这称为“零填充”，以便设计矩阵具有与尖峰中的响应矢量相同的行数。  
您的任务是完成以下功能： 
- 制作刺激的零填充版本 
- 使用刺激的零填充版本初始化一个具有正确形状的空设计矩阵
- 填充设计矩阵的每一行
zero-padding: np.zeros(d - 1) 生成了一个长度为 d-1 的零数组，然后这个数组和 stim 被连接在一起，形成了一个新的一维数组 padded_stim。  
对于每个时间点 i，X[i, j] 对应的是从当前时间点 i 开始，向前 j 个时间点的刺激强度。例如，X[i, 0] 就是在时间点 i 及其前 d-1 个时间点的刺激强度，X[i, d-1] 是在时间点 i 的刺激强度。  
## section 1.2 Fit Linear-Gaussian regression model、
为了模型能够学习到数据的偏置项，我们经常会在设计矩阵（X）中增加一个常数列，通常是全1列。在这个具体的上下文中，你的目标是找到最大似然估计值θ，使得模型y = Xθ尽可能地拟合数据。在这个公式中，X是设计矩阵，y是观测值，θ是模型参数。这是一个线性模型，但如果没有截距项（或者说偏置项b），那么这个模型将被限制在原点通过
plot_spike_filter(theta_lg, dt_stim)绘制刺激滤波器的估计值，这是线性高斯广义线性模型（linear-Gaussian GLM）中的参数θ。这个滤波器描述了刺激如何影响神经元的反应
在神经系统中，我们经常发现近期的输入对输出有更大的影响，而过去的输入影响力则逐渐衰减。这就是为什么我们经常看到模型的权重随着时间滞后的增加而减小
### Coding Exercise 1.2: Predict spike counts with Linear-Gaussian model
- Create the complete design matrix
- Obtain the MLE weights ( θ^ )
- Compute  y^=Xθ^

 "spike-triggered average"（STA）和线性高斯广义线性模型（Linear Gaussian Generalized Linear Model, LG GLM）是两种估计神经元反应的方法之间的关系：  
STA是一种较为简单的方法，它计算的是在观察到脉冲（或"spike"）时，前一段时间内的平均刺激。数学上，这可以表示为 STA = X.T @ y / sum(y)，其中 X 是设计矩阵，y 是神经元的脉冲计数向量。  然而，STA的一个缺点是，它假设设计矩阵 X 的各个列之间（即不同时间滞后的刺激）是互相独立的，即它们之间没有相关性。这在实际情况中可能不成立，特别是当刺激之间存在一定的时间相关性时。  相比之下，LG GLM通过引入一项 (X.T @ X)−1 来纠正这种可能的相关性。这样，即使在刺激之间存在相关性的情况下，LG GLM也能得到更为准确的估计。然而，在本实验中，所使用的刺激是"白噪声"，即各个时间点的刺激是互相独立的。因此，不存在刺激之间的相关性，STA和LG GLM的结果应该是一致的。关于如何检查刺激之间没有相关性，一种常见的方法是计算设计矩阵 X 各列之间的相关系数（或协方差），并检查其是否接近0。在numpy中，可以使用 numpy.corrcoef 或 numpy.cov 函数来计算相关系数或协方差。如果各列之间的相关性接近于0，那么就可以认为刺激之间不存在相关性。
## Section 2.1: Nonlinear optimization with scipy.optimize
线性-高斯模型（Linear-Gaussian model）下，最大似然参数估计可以解析地求解。然而通常情况下，我们需要解决的统计估计问题并没有解析解，我们需要应用非线性优化算法来寻找使某些目标函数最小化的参数值，而我们不知道是否找到了全局最优解还是局部最优解，。如果目标函数是凸函数，这类优化问题可以非常可靠地解决  
凸函数的特点是，其曲线在连接其任意两点的线段之下
### Coding Exercise 2.1: Fitting the Poisson GLM and prediction spikes
在LNP模型中，对一个给定的输入，首先进行线性滤波（这是由X @ theta实现的），然后通过一个非线性函数（这里是指数函数exp）转化，最后假设输出服从泊松分布。yhat = np.exp(X@theta)是非线性阶段的实现. X@theta实现了线性滤波，输出是神经元的驱动力（drive），然后np.exp函数将这个驱动力转化为预期的神经元的发射频率（firing rate）。  
往往假设神经元的firing rate是非负的，并且随着驱动力的增加而增加。指数函数是一个常用的选择.yhat是预测的平均firing rate,真实的firing rate可能因为各种噪声的影响而有所波动。在LNP模型中，这种波动通常被假设为泊松分布。
## Summary
在这个教程中，我们探索了两种不同的模型（Linear-Nonlinear-Poisson (LNP)模型和Linear-Gaussian (LG)模型）来理解视网膜神经节细胞如何响应闪烁的白噪声刺激。
  我们学习了如何构建设计矩阵（design matrix），这是一个可以输入到各种广义线性模型（GLMs）的重要数据结构。设计矩阵的每一行对应于一种特定的刺激情况，并包含该时间点之前的刺激强度信息，这对于理解神经元是如何处理时间相关信息的非常有用。  
通过比较LNP和LG模型，我们发现LNP模型能更好地预测神经元的尖峰率。这是因为LNP模型对神经元反应的非线性和随机性有更好的处理，这反映了神经元的实际行为。然而，LNP模型也比LG模型更复杂，需要更复杂的优化方法来估计模型参数。  

# Tutorial 2: Classifiers and regularizers
