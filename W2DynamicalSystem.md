# W2D2:
## Tutorial 1: Linear dynamical systems
线性常微分方程的近似解：Forward Euler Method  
Section 2: Oscillatory Dynamics  
when  a  is a complex number and has a non-zero imaginary component.  
同样的，X(t)=exp(αt) xdot=αx
可以分解为x_0*exp[(at)(cosbt+isinbt)]形式,通过尝试改变实部，虚部正负，可以得到稳定振荡，发散，收敛几种结果  
Section 3: Deterministic Linear Dynamics in Two Dimensions  
 simulate some trajectories of a given system and plot how 𝑥1 and 𝑥2 evolve in time. 
Section 4: Stream Plots  
流线图：更直观地展示一组初始条件是如何影响系统轨迹  
初始条件作为空间坐标: 可以将初始条件X_0视为空间中的一个位置。在这个上下文中，初始条件是一个点，表示系统开始演化的起点。  
对于一个2x2矩阵 A，流线图在每个位置 x 计算一个小箭头来指示 Ax，然后连接这些小箭头形成流线。这里 Ax 表示系统在该位置的变化方向和速度。穿过该点的流线表示 x(t)，即系统从该初始条件随时间演化的轨迹。  
主特征向量指向的方向特殊之处在于，它是沿着该方向 Ax 与 x 平行，并且在该方向上 Ax 相对于 x 被拉伸或压缩的最大因子。在许多情况下，这表明了系统的主要变化方向或敏感方向。  
系统的稳定性与特征值有关。如果特征值大于1，系统在特征向量的方向上展开；如果特征值小于1，系统在该方向上收缩；如果特征值为负，则系统在该方向上倒转。稳定性与这些特征值的大小和符号有关，可以帮助我们理解系统在不同方向上的行为如何组合以产生整体的动态行为。  
Summary:  模拟动力系统的轨迹
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

Section 2: Distributional Perspective  
函数simulate_prob_prop，该函数模拟了给定转换矩阵A的概率传播，初始状态x0，总持续时间T和时间步长dt。这个函数实现了一个离散时间马尔可夫过程，可以用于模拟像离子通道这样的系统的打开和关闭状态
使用np.dot(A,x[-1,:])时，x作为一个行向量被numpy自动解释成了列向量  
Section 3: Equilibrium of the telegraph process  
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
