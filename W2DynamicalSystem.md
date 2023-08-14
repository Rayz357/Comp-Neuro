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
