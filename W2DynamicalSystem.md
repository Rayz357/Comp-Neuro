# W2D2:
## Tutorial 1: Linear dynamical systems
线性常微分方程的近似解：Forward Euler Method  
### Section 2: Oscillatory Dynamics  
when  a  is a complex number and has a non-zero imaginary component.  
同样的，X(t)=exp(αt) xdot=αx
可以分解为x_0*exp[(at)(cosbt+isinbt)]形式,通过尝试改变实部，虚部正负，可以得到稳定振荡，发散，收敛几种结果  
### Section 3: Deterministic Linear Dynamics in Two Dimensions  
 simulate some trajectories of a given system and plot how 𝑥1 and 𝑥2 evolve in time. 
### Section 4: Stream Plots  
流线图：更直观地展示一组初始条件是如何影响系统轨迹  
初始条件作为空间坐标: 可以将初始条件X_0视为空间中的一个位置。在这个上下文中，初始条件是一个点，表示系统开始演化的起点。  
对于一个2x2矩阵 A，流线图在每个位置 x 计算一个小箭头来指示 Ax，然后连接这些小箭头形成流线。这里 Ax 表示系统在该位置的变化方向和速度。穿过该点的流线表示 x(t)，即系统从该初始条件随时间演化的轨迹。  
主特征向量指向的方向特殊之处在于，它是沿着该方向 Ax 与 x 平行，并且在该方向上 Ax 相对于 x 被拉伸或压缩的最大因子。在许多情况下，这表明了系统的主要变化方向或敏感方向。  
系统的稳定性与特征值有关。如果特征值大于1，系统在特征向量的方向上展开；如果特征值小于1，系统在该方向上收缩；如果特征值为负，则系统在该方向上倒转。稳定性与这些特征值的大小和符号有关，可以帮助我们理解系统在不同方向上的行为如何组合以产生整体的动态行为。  
### Summary:  模拟动力系统的轨迹
How to simulate the trajectory of a dynamical system specified by a differential equation  x˙=f(x)  using a forward Euler integration scheme.  
The behavior of a one-dimensional linear dynamical system  x˙=ax  is determined by  a , which may be a complex valued number. Knowing  a , we know about the stability and oscillatory dynamics of the system.  
The dynamics of high-dimensional linear dynamical systems  x˙=Ax  can be understood using the same intuitions, where we can summarize the behavior of the trajectories using the eigenvalues and eigenvectors of  A .  
一维线性动态系统随时间变化的轨迹用欧拉方法模拟，高维线性动态系统(x˙=Ax)用矩阵表示后，可以用流线图结合特征值特征向量绘制其轨迹。通过矩阵 A 的特征分解，我们可以了解系统在各个方向上的行为，并使用这些知识来分析和控制动力学系统
# Tutorial 2: Markov Processes
In Tutorial 1, we studied dynamical systems as a deterministic process. For Tutorial 2, we will look at probabilistic dynamical systems.马尔可夫过程是概率过程的一种特殊类型。在马尔可夫过程中，下一状态的转换完全由当前状态决定，与过去的状态无关。
泊松过程  
泊松过程被用来模拟状态改变的过程。泊松过程用于模拟离散事件，其中平均事件发生间隔时间已知，但某个具体事件的确切时间未知。泊松过程的几个关键点包括：

某个事件发生的概率与所有其他事件独立。
给定时间段内的事件平均速率是恒定的。
两个事件不能同时发生（离子通道不能同时处于打开和关闭状态）。  

### Section 2: Distributional Perspective  
函数simulate_prob_prop，该函数模拟了给定转换矩阵A的概率传播，初始状态x0，总持续时间T和时间步长dt。这个函数实现了一个离散时间马尔可夫过程，可以用于模拟像离子通道这样的系统的打开和关闭状态
使用np.dot(A,x[-1,:])时，x作为一个行向量被numpy自动解释成了列向量  
### Section 3: Equilibrium of the telegraph process  
特征分解（eigendecomposition）是一个强大的数学工具，可用于分析线性动态系统的稳定性和行为。如何通过分解转移矩阵 A 的特征值和特征向量来了解系统的性质。  
给定一个方阵 A，我们可以找到一些向量 v 和标量 λ，满足：Av=λv这里的 v 称为特征向量，λ 称为对应的特征值。  
特征值的大小和符号提供了关于系统稳定性的信息： 
- 如果特征值的模小于 1（|λ| < 1），那么随着时间的推移，系统沿着对应的特征向量方向的分量会衰减到零。这表明系统是稳定的。
- 如果特征值的模等于 1（|λ| = 1），系统沿着该方向的分量将保持不变。这可能是一个中性稳定的方向。
- 如果特征值的模大于 1（|λ| > 1），系统沿着该方向的分量将随时间增长。这表明系统在该方向上是不稳定的。
### Summary：
- The definition of a Markov process with history dependence.
- 电报过程是一个描述两个状态之间随机转换的模型。例如，这可以模拟离子通道的开闭状态。你可以用两种方式模拟电报过程：一是作为状态更改的模拟，即在每个时间点确定状态的更改；二是作为概率分布的传播，即通过转换矩阵跟踪状态概率的变化。对于第一种方式，我们使用了泊松过程来模拟离子通道的状态，可以在整个模拟时间内观察通道的状态，并测量状态切换之间的时间间隔分布。状态更改的模拟可以在给定的时间步内，根据转换概率，确定系统是否从一个状态转换到另一个状态（也就是给出状态-时间关系图）。与第一种方法不同，概率分布的传播不关心系统的具体实现，而是跟踪系统状态的概率分布
- The relationship between the stability analysis of a dynamical system expressed either in continuous or discrete time.在连续时间动态系统中，稳定性可以通过系统的特征值来分析。特征值的实部确定了系统沿着相应特征向量方向的增长或衰减。离散时间系统也具有相似的分析，其中特征值的模的大小决定了沿着特征向量方向的行为。连续和离散时间系统的主要区别在于时间的表示和演变方式，但稳定性分析的基本思想在两者之间保持一致
- The equilibrium behavior of a telegraph process is predictable and can be understood using the same strategy as for deterministic systems in Tutorial 1: by taking the eigendecomposition of the  A  matrix.
## Tutorial 3: Combining determinisim and stochasticty
### Section 1: Random Walks
随机漫步是一种简单的随机过程，其中一个“行走者”在每一步都随机地选择方向。虽然每一步的方向是随机的，但随着时间的推移，我们可以观察到某些统计特性。这是一个很好的理解随机性如何工作的入门示例。
- 在随机漫步中，由于向左和向右的选择概率相等，所以移动的期望值是零
- 虽然均值不随时间变化，但方差确实随时间线性增加。方差衡量了分布的宽度，或者说位置的不确定性。在随机漫步中，由于每一步都是完全随机的，所以位置的不确定性随时间线性增加。这可以数学上表示为 Var(x)∝t，
这样的过程称为扩散过程
### Section 2: The Ornstein-Uhlenbeck (OU) process
漂移扩散模型（Drift-Diffusion Model, DDM），这是一个将随机漫步过程与确定性动态过程相结合的模型    
x_(k+1)=λx_k 其中 x(k) 是在时间 k 的状态，而 λ 是一个常数，控制着系统如何随时间演变。  
该方程的解为：x(k) = x(0) * λ^k
Notice that this process decays towards position  x=0 . We can make it decay towards any position by adding another parameter  x∞ . The rate of decay is proportional to the difference between  x  and  x∞ . Our new system is  
xk+1=x∞+λ(xk−x∞) OR xk=x∞(1−λk)+x0λk.
#### 随机漫步和漂移扩散模型
- 随机漫步可以看作是DDM的一个特例，当漂移项为零时。
- DDM通过引入一个稳定的漂移项，为随机漫步过程增加了方向性和预测性。
- 漂移项提供了一个平均方向，而扩散项则描述了围绕这个平均方向的随机波动。

### Section 3: Variance of the OU process
### Summary
确定性和随机部分：OU系统包括一个确定性的部分和一个随机的部分。确定性部分通常描述了系统的平均行为或趋势，而随机部分描述了由于未知或不可预测因素造成的偏离这些平均行为的波动。  
与纯随机过程的比较：与纯随机过程（例如随机漫步）相比，OU系统的方差不会随时间无限增长。确定性部分对随机波动有所约束，从而使系统保持更有序和可预测的状态。  
在认知功能中的应用：由于OU系统可以平衡随机性和确定性，它们成为了许多认知功能模型的热门选择。例如，在短时记忆和决策制定中，人们需要考虑固定的规则和可预测的模式，同时还需要灵活地适应新的信息和不确定的情况。OU系统通过结合这两个方面为这些过程提供了一个有效的数学框架。
## Tutorial 4: Autoregressive models
第一部分 遵循使用回归对数据进行处理，以从第3个教程的OU过程中解出系数的方法。OU过程（Ornstein-Uhlenbeck过程）是一种结合了确定性和随机部分的动态系统，可以用于模拟许多自然现象。
第二部分 将这个自回归框架推广到高阶自回归模型，并尝试拟合来自“猴子在打字机上打字”的数据。这是一个比喻，通常用来形容随机过程，或者形容在无规律、随机输入的情况下产生的结果。
### Section 1: Fitting data to the OU process
这一节一阶自回归模型（first-order autoregressive model）的设置和求解过程。自回归模型是时间序列分析中常用的一种模型，它假设当前时刻的观察值与过去一段时间内的观察值之间存在线性关系。在一阶自回归模型中，当前时刻的值仅与前一时刻的值有关
### Section 2: Higher order autoregressive models
高阶自回归模型能够捕捉时间序列中更复杂的时间依赖结构，使模型能够更精确地描述数据的动态行为。不过，选择合适的阶数是一个挑战，因为阶数太低可能无法充分捕捉依赖性，而阶数太高可能导致过拟合。在实际应用中，人们通常会使用诸如AIC或BIC等信息准则来确定最佳的阶数。
xk+1=α0+α1xk+α2xk−1+α3xk−2+⋯+αr+1xk−r  

# W2D3: Biological Neuron Model
## Tutorial 1:The Leaky Integrate-and-Fire (LIF) Neuron Model
这个教程的目标是通过对LIF神经元模型的研究，提供对神经元如何响应各种输入的理解，特别是关注在哪些条件下神经元可能以低频率和不规则的方式发射
1. 模拟LIF神经元模型
2. 用外部输入驱动LIF神经元
3. 研究不同输入对LIF神经元输出的影响
4. 重点关注不规则和低频率发射的条件
### Section 1:
LIF模型的特点
- 执行突触输入的空间和时间积分
- 当电压达到某一阈值时产生脉冲
- 在动作电位期间进入不应期
- 具有漏电膜
LIF模型的假设
- 空间和时间输入的积分是线性的。
- 接近动作电位阈值的膜电位动态在LIF神经元中比在真实神经元中慢得多。
### Section 2: Response of an LIF model to different types of input currents
- 达到阈值所需的直流电流有多少?
- 膜时间常数如何影响神经元的频率？
-   膜时间常数（τ_m）是电位达到其稳态值的时间量度。较大的膜时间常数意味着电位变化较慢，因此可能需要更长的时间才能达到阈值，
- 噪声如何影响神经元行为?通过引入高斯白噪声进行模拟
-   更大的电流波动（增加的σ）降低了发放阈值：当σ增加时，需要的μ减小
-   标准偏差σ（或电流波动的大小）控制了脉冲的不规则性水平
- GWN的均值、方差与脉冲计数的关系
-   较高的均值和方差都会增加脉冲计数，并造成不规则性。不应期等生理属性会限制不规则形的最大值
### Section 3: Firing rate and spike time irregularity  
第3节介绍了如何通过GWN（高斯白噪声）均值或DC值来绘制神经元的输出发射率，称为神经元的输入-输出传递函数（脉冲个数-平均输入电流），或简称为F-I曲线。还介绍了如何通过计算interspike interval（ISI）的变异系数（CV）来量化脉冲的规律性。
- 脉冲时间间隔的变异系数（CVISI）：CVISI是衡量脉冲时间不规则性的量。计算为ISI的标准偏差除以ISI的均值。CVISI的值为1表示高不规则性，如泊松过程；值为0表示完全规则的过程。
- DC输入：在直流输入下，神经元的响应是确定的。也就是说，对于给定的输入电流强度，总是会得到相同的输出频率。F-I曲线通常可能是非线性的，依赖于神经元和输入的特定属性。通过解神经元的膜方程，可以精确地找到这个曲线的形状。
- GWN输入：当使用高斯白噪声输入时，情况变得更复杂。噪声的平均值可以视为一个DC分量，但噪声的标准偏差（或sigma）增加了不确定性和随机性。随着sigma的增加，F-I曲线变得更线性。由于噪声的随机波动，神经元可能会以较低的平均注入电流达到阈值。这样的效应可以导致更灵活的编码和响应，但也使行为变得随机和不可预测。
- CV（系数变异，Coefficient of Variation）用于衡量一个数据系列的标准偏差与均值之间的关系，在这里用于衡量脉冲间隔时间（Interspike Interval，ISI）的不规则性
** CV of inter-spike-interval (ISI) distribution depends on  σ  of GWN**
  GWN的均值和标准差是如何影响ISI分布的？
- 当神经元被直流（DC）输入驱动时，频率-电流（F-I）曲线显示出强烈的非线性。但是，当用高斯白噪声（GWN）驱动时，随着噪声强度（标记为 σ）的增加，非线性被抚平。这里的基本思想是噪声在某种程度上充当了一个均衡因素，减弱了系统的非线性效应，使神经元更接近线性系统。
- 高GWN均值导致高发放率，并降低了脉冲间隔的变异系数（CV_ISI）。这意味着随着发放率的增加，脉冲的不规则性减少。

### Summary: 
- 模拟了LIF神经元模型
- 使用外部输入（如直流和高斯白噪声）驱动LIF神经元
- 研究了不同输入如何影响LIF神经元的输出（发放率和脉冲时间不规则性）
- 特别关注了低发放率和不规则发放模式，以模拟真实的皮层神经元。
## Tutorial 2: Effects of Core
研究神经元是如何将输入相关性转化为输出特性（相关性的传递）。具体来说，我们将编写一些代码来：
- 在一对神经元中注入相关的GWN（高斯白噪声）
- 测量两个神经元之间的脉冲活动的相关性
- 研究相关性传递如何依赖于输入的统计特性，即均值和标准差。

### Section 1: Correlations (Synchrony)
相关性/同步性是指神经元的同时脉冲,有以下几种原因：
- 共同输入：即两个神经元从相同的源接收输入。共享输入的相关度与它们的输出相关度成正比。
- 从相同源池化：神经元不共享相同的输入神经元，但从彼此相关的神经元接收输入。
- 神经元彼此连接（单向或双向）：这只会引起时间延迟同步。神经元也可以通过间隙连接相连。
- 神经元具有相似的参数和初始条件。

同步性的影响：
- 当神经元一起脉冲时，它们可以对下游神经元产生更强的影响。大脑中的突触对突触前和突触后活动之间的时间相关性（即延迟）敏感，这反过来可以导致功能神经网络的形成 - 无监督学习的基础
- 同步性暗示了系统维度的减少。此外，相关性在许多情况下可能会损害神经元活动的解码。


在下一个练习中，我们将使用泊松分布来模拟脉冲列。记得你在统计学预备课程中已经看到泊松分布以这种方式被使用过。要记得泊松脉冲列具有以下特性：
- 脉冲计数的均值和方差之比为1。
- 脉冲间隔呈指数分布。
- 脉冲时间不规则，即 CV_ISI=1。
- 相邻的脉冲间隔彼此独立。
我们提供一个辅助函数Poisson_generator，然后使用它生成一个泊松脉冲列。Poisson spike train是一种数学模型，用于描述神经元（或一组神经元）的发放活动。在这个模型中，神经元在每个给定的时间间隔内发放一个动作电位（或“脉冲”）的概率是常数。该模型的名称用于描述脉冲发生的随机性。

### Section 2: Investigate the effect of input correlation on the output correlation
We first generate the correlated inputs. Then we inject the correlated inputs  I1,I2  into a pair of neurons and record their output spike times. We continue measuring the correlation between the output and investigate the relationship between the input correlation and the output correlation.  

- 输出相关性小于输入相关性
- 输出相关性作为输入相关性的函数呈线性变化。
- 这些结果展示了输入信号在通过神经元传输过程中的转换特性。

The above plot of input correlation vs. output correlation is called the correlation transfer function of the neurons.相关性传递函数为我们提供了一种量化神经元或神经元网络如何响应不同程度的输入相关性的方法。
### Section 3: Correlation transfer function
How do the mean and standard deviation of the Gaussian white noise (GWN) affect the correlation transfer function?  
**线性F-I曲线**：如果F-I曲线是线性的，输出相关性就与输入的均值和标准偏差无关。换句话说，输入电流的任何部分都可以直接转换为突触活动，而不会发生失真。  
**非线性F-I曲线**：然而，实际的神经元通常具有非线性的F-I曲线。即使是具有阈值线性F-I曲线的神经元，输入电流的均值和标准偏差也会影响输出相关性。这可能是因为神经元对于超过一定阈值的输入更敏感，因此只有超过该阈值的输入才能有效地转化为输出突发。此外，输入电流的均值和标准偏差可能会决定神经元是在其F-I曲线的哪个部分操作。  
What is the rationale behind varying  μ  and  σ ?  
神经突触电流的均值和方差取决于Poisson过程的尖峰速率。我们可以使用被称为Campbell定理的东西来估计突触电流的均值和方差：  
μ_syn = λJ∫P(t)dt    
σ_syn = λJ∫P(t)^2dt  
其中，λ是Poisson输入的发射速率，J是突触后电流的幅度，P(t)是突触后电流作为时间函数的形状。  
因此，当我们改变GWN的μ和/或σ时，我们模拟了输入发射速率的变化。请注意，如果我们改变发射速率，μ和σ将同时改变，而不是独立改变。  
1. What are the factors that would make output correlations smaller than input correlations? (Notice that the colored lines are below the black dashed line)
   - 转移过程中减小相关性的可能机制：什么因素可能使输出相关性小于输入相关性。平均值和方差的减小、神经元之间参数的差异，以及神经传递函数的斜率都可能导致输入和输出之间的相关性降低。
2. What does the fact that output correlations are smaller mean for the correlations throughout a network?
   - 如果输出相关性始终小于输入相关性，那么理论上网络活动的相关性最终应该趋近于零。但在实际情况中，这并没有发生，所以当前模型中似乎缺少了一些东西，这些因素可能是理解网络同步起源的关键。
3. Here we have studied the transfer of correlations by injecting GWN. But in the previous tutorial, we mentioned that GWN is unphysiological. Indeed, neurons receive colored noise (i.e., Shot noise or OU process). How do these results obtained from injection of GWN apply to the case where correlated spiking inputs are injected in the two LIFs? Will the results be the same or different?
   - 当考虑脉冲输入而不是高斯白噪声输入时结果是否相同。作者认为结果在定性上将是相似的，因为脉冲输入的火率与输入的均值和方差有关。但是，当考虑多个脉冲输入时，可能会产生两种不同类型的相关性，需要进一步研究和理解。
