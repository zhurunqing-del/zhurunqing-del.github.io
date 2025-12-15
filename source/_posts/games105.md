---
title: games105
date: 2025-09-25 12:55:10
tags: ["games", "图形学"]
katex: true
---
games105笔记以及作业。


## 01 Introduce to Character Animation
### 什么是3d动画
动画、游戏、vr、数字人、大规模人物动作仿真

3d计算机图形学
![动画](/image/games105/截图_20251110221248.png)

3d计算机动画
仿真+角色动画
仿真：刚体、可变形固体、绳索、薄壳、布料、流体、烟雾、声音……现象

角色动画：行为 人类、动物、虚拟生物、手、机器人、人群…
![alt text](/image/games105/截图_20251110221522.png)

### 为什么学习3d动画
* 一个角色通常有20多个关节，或50-100多个参数
    * 它的维度不是特别高，所以大部分动画可以通过在关键帧上摆姿势手动创建 
    * 劳动密集型，不适合交互式应用
* 角色动画技术 
    * 理解动作和行为背后的机制 
    * 智能编辑动画 / 重用动画 / 生成新动画 
    * “计算密集型”

角色动画管线
![alt text](/image/games105/截图_20251110222006.png)

（基于物理/动态的方法与基于关键帧/运动学的方法）
低级别控制--高级别控制
![alt text](/image/games105/截图_20251110222508.png)

关键帧动画
![alt text](/image/games105/截图_20251110222734.png)

正向运动学
给定每个关节的旋转，计算末端执行器的位置

逆向运动学
给定末端执行器的位置，计算每个关节的旋转角度

差值动画
### 角色动画方法
**动捕**
多视角相机+标记点
![alt text](/image/games105/截图_20251111210629.png)
惯性动捕
![alt text](/image/games105/截图_20251111210703.png)
视频动捕
![alt text](/image/games105/截图_20251111210729.png)

**重定向**
在给定源角色动作的情况下，为目标角色计算动作，
要求目标角色具有：
• 不同的骨架尺寸
• 不同的骨骼数量
• 不同的拓扑结构

**运动图 / 状态机**
![alt text](/image/games105/截图_20251111210821.png)
论文：Motion Graphs
训练ai使用motiongraph
![alt text](/image/games105/截图_20251111211352.png)
复杂运动图
![alt text](/image/games105/截图_20251111211518.png)

**动作匹配（Motion Matching）**
每一帧动作进行切片

**动作生成模型**
从数据中学习（深度学习）
![alt text](/image/games105/截图_20251111212049.png)
![alt text](/image/games105/截图_20251111212653.png)

**跨模态运动合成**
音频驱动动画 
• 音乐舞蹈 
• 语音同步手势
自然语言到动画 
• 描述到动作 
• 剧本到表演

### 运动学方法的问题
**物理可行性**
![alt text](/image/games105/截图_20251111213057.png)
**与环境的互动**
![alt text](/image/games105/截图_20251111213155.png)

### 基于物理/动态方法
**关键帧控制**
![alt text](/image/games105/截图_20251111213337.png)
* 布偶模拟
![alt text](/image/games105/截图_20251111213428.png)

* 基于物理/动态方法
![alt text](/image/games105/截图_20251111213610.png)

* 力与扭矩 
![alt text](/image/games105/截图_20251111213725.png)

* 比例-微分 (PD) 控制
![alt text](/image/games105/截图_20251111213855.png)
两个高度差乘系数，就是需要的力
![alt text](/image/games105/截图_20251111214007.png)

* 轨迹设计
![alt text](/image/games105/截图_20251111214252.png)

**时空/轨迹优化**
![alt text](/image/games105/截图_20251111214750.png)
![alt text](/image/games105/截图_20251111214813.png)

**抽象模型**
线性反馈控制 有限状态机（常用于机器人）
![alt text](/image/games105/截图_20251111215334.png)

**基于深度强化学习的跟踪控制器**
![alt text](/image/games105/截图_20251111215528.png)

**分层控制器Hierarchical Controllers**

**生成控制策略**
![alt text](/image/games105/截图_20251111215911.png)
![alt text](/image/games105/截图_20251111220045.png)

### 角色动画控制方法
![alt text](/image/games105/截图_20251111220113.png)

## 02 Math Background
### 线性代数复习
**向量**
向量的长度、方向、标准向量
向量同时拥有大小和方向，可以用来表示位置速度或更高维度的特征
向量的计算（加法，乘法）
* 高维向量的表示
$
a= \left ( a{\tiny 1},a{\tiny 2},\dots ,a{\tiny n} \right )^{T}   =\begin{bmatrix}
 a{\tiny 1}\\a{\tiny 2}\\\vdots 
 \\a{\tiny n}
\end{bmatrix}
$
长度、大小
$
\left \| a \right \| _{2} = \sqrt{a_{1}^{2} + a_{2}^{2}+\dots +a_{n}^{2}}
$

三维向量的表示
![alt text](/image/games105/截图_20250926111931.png)
$a= \left ( a{\tiny 1},a{\tiny 2},a{\tiny 3} \right )^{T}   =\begin{bmatrix}
 a{\tiny 1}\\a{\tiny 2}
 \\a{\tiny 3}
\end{bmatrix}$

长度、大小
$
\left \| a \right \| _{2} = \sqrt{a_{x}^{2} + a_{y}^{2}+a_{z}^{2}}
$

* 点乘
$
a\cdot b =a_{1} b_{1}  +a_{2} b_{2} +\cdots +a_{n} b_{n} 
$

$
a\cdot b =b\cdot a
$

$
a\cdot \left ( b+c  \right ) =a\cdot b+b\cdot c
$

$
a\cdot a = a_{1} a_{1} +a_{2} a_{2} +\cdots +a_{n} a_{n}  =\left \|  a \right \| _{2} ^{2} 
$

计算角度
计算投影

* 叉乘

![叉乘](/image/games105/截图_20250926123057.png)
$
c = a\times b = \begin{bmatrix}
 a_{y} b_{z} -a_{z} b_{y} \\a_{z} b_{x} -a_{x} b_{z} 
 \\a_{x} b_{y} -a_{y} b_{x} 
\end{bmatrix}
$

$
c = a\times b = \left \| a \right \| \left \| b \right \| \sin \left ( \theta  \right ) n
$

* 计算最小旋转

![alt text](/image/games105/截图_20250926123945.png)

![alt text](/image/games105/截图_20250926130347.png)
由于u是单位向量，所以 $\left \| u\times a \right \| $  是到a到u的距离

$
b = a+\left ( \sin\left ( \theta  \right ) u \times a +\left ( 1-\cos \theta  \right ) u\times \left ( u\times a \right )   \right )
$

* 正交基与正交坐标
点乘：
![alt text](/image/games105/截图_20250926150438.png)
叉乘：
![alt text](/image/games105/截图_20250926150608.png)

**矩阵**
* 定义
二维数组

* 特殊矩阵
单位矩阵 $A^{\mathrm{T}}=A$
$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$

对角矩阵 $A^{\mathrm{T}}=A$
$
\begin{bmatrix}
a_1 & 0 & 0 \\
0 & a_2 & 0 \\
0 & 0 & a_3
\end{bmatrix}
$

对称矩阵 $A^{\mathrm{T}}=A$
$
\begin{bmatrix}
a & d & e \\
d & b & f \\
e & f & c
\end{bmatrix}
$
斜对称矩阵 $A^{\mathrm{T}}=-A$
$
\begin{bmatrix}
0 & -a & b \\
a & 0 & -c \\
-b & c & 0
\end{bmatrix}
$

* 转置
$
\begin{gathered}
A=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}\in\mathbb{R}^{3\times3} \\
=
\begin{bmatrix}
a_1 & a_2 & a_3
\end{bmatrix}=
\begin{bmatrix}
a_{1*} \\
a_{2*} \\
a_{3*}
\end{bmatrix}
\end{gathered}
$
$
A^\mathrm{T}=
\begin{bmatrix}
a_{11} & a_{21} & a_{31} \\
a_{12} & a_{22} & a_{32} \\
a_{13} & a_{23} & a_{33}
\end{bmatrix}=
\begin{bmatrix}
a_1^T \\
a_2^T \\
a_3^T
\end{bmatrix}=
\begin{bmatrix}
a_{1*}^T & a_{2*}^T & a_{3*}^T
\end{bmatrix}
$

* 矩阵的加减法乘法
* 矩阵与矩阵的乘法
$
\begin{aligned}& A=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}\in\mathbb{R}^{3\times3} \\
C=AB & =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{bmatrix} \\
& =
\begin{bmatrix}* & ? & * \\* & * & * \\* & * & *
\end{bmatrix}\quad=
\begin{bmatrix}
c_{ij}=a_{i*}\cdot b_j
\end{bmatrix}
\end{aligned}
$

* 矩阵的性质（不满足交换律，满足结合律，逆矩阵）
$
\begin{aligned}
\boxed{AB\neq BA} \\
 & ABC=(AB)C=A(BC) \\
 & A(B+C)=AB+AC \\
 & (AB)^\mathrm{T}=B^\mathrm{T}A^\mathrm{T}\quad IA=A
\end{aligned}
$
$
\begin{aligned}
 & M=A^{-1}\Leftrightarrow AM=MA=I \\
 & (AB)^{-1}=B^{-1}A^{-1}
\end{aligned}
$

* 点乘性质（从矩阵理解）
$
\begin{aligned}
\boldsymbol{a\cdot b} & =a_xb_x+a_yb_y+a_zb_z \\
 & =a^Tb=
\begin{bmatrix}
a_x & a_y & a_z
\end{bmatrix}
\begin{bmatrix}
b_x \\
b_y \\
b_z
\end{bmatrix} \\
 & =\boldsymbol{b}^T\boldsymbol{a}
\end{aligned}
$

* 叉乘性质（从矩阵理解）
$
\begin{aligned}
\boldsymbol{c}=\boldsymbol{a}\times\boldsymbol{b} & =
\begin{bmatrix}
a_yb_z-a_zb_y \\
a_zb_x-a_xb_z \\
a_xb_y-a_yb_x
\end{bmatrix} \\
 & =
\begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}
\begin{bmatrix}
b_x \\
b_y \\
b_z
\end{bmatrix} & =[\boldsymbol{a}]_\times\boldsymbol{b}
\end{aligned}
$
$
[a]_\times+[a]_\times^T=0 
$
skew_symmetric
$
\begin{aligned}
a\times b & =[a]_\times\boldsymbol{b} \\
a\times(b\times c) & =[a]_\times([b]_\times c) \\
 & =[a]_\times[b]_\times c \\
a\times(a\times c) & =[a]_\times^2\boldsymbol{b}
\end{aligned}
$
$
(a\times b)\times c=[a\times b]_\times c
$

* 旋转（从矩阵理解）
![alt text](/image/games105/截图_20251004212924.png)
![alt text](/image/games105/截图_20251004213004.png)

* 正交矩阵
一个列（及行）为正交向量的矩阵
![alt text](/image/games105/截图_20251004214008.png)

* 矩阵的行列式
左上到右下三个乘积和减去右上到左下三个乘积的和
![alt text](/image/games105/截图_20251111231123.png)
![alt text](/image/games105/截图_20251111231337.png)

叉积作为行列式
![alt text](/image/games105/截图_20251111231842.png)


* 特征值
对于矩阵𝐴，如果一个非零向量𝒙满足
𝐴𝒙 =𝜆𝒙
𝜆：𝐴 的一个特征值  
𝒙：𝐴 的一个特征向量
特别地，一个 3 × 3 正交矩阵 𝑈 至少有一个实特征值：𝜆 = det𝑈 = ±1

**变换**
平移、旋转、缩放

* 旋转矩阵
旋转矩阵是正交的：
$
R^{-1}=R^\mathrm{T}\quad R^\mathrm{T}R=RR^\mathrm{T}=I
$
行列式：
det𝑅 = +1
旋转保持向量的长度
$
\|Rx\|=\|x\|
$
旋转矩阵 𝑅 有一个实特征值：±1
𝑅𝒖 =𝒖
换句话说，𝑅 可以被认为是绕轴 𝒖 按某个角度θ旋转。
![alt text](/image/games105/截图_20251112084042.png)
𝒖′ ×𝒖=0
$
u\leftarrow u^{\prime}=
\begin{bmatrix}
r_{32}-r_{23} \\
r_{13}-r_{31} \\
r_{21}-r_{12}
\end{bmatrix}
$
![alt text](/image/games105/截图_20251112084749.png)
![alt text](/image/games105/截图_20251112084837.png)
![alt text](/image/games105/截图_20251112085003.png)

* 坐标系转换
![alt text](/image/games105/截图_20251112085543.png)
![alt text](/image/games105/截图_20251112085608.png)
![alt text](/image/games105/截图_20251112085902.png)

### 3d旋转
**旋转参数化**
6个参数与其他参数互相不独立的，只有3个参数互相独立，
因此自由度为3（需要多少参数才能描述一个物体）
![alt text](/image/games105/截图_20251112090248.png)

**旋转插值**
什么是良好的插值？
• 旋转在任意时间 𝑡 都是有效的 
• 需要恒定的旋转速度
![alt text](/image/games105/截图_20251112091207.png)

**旋转表达方式**
* 旋转矩阵
![alt text](/image/games105/截图_20251112091547.png)
• 容易组成吗？ ×
• 容易应用吗？ √
• 容易插值吗？ ×

* 欧拉角
![alt text](/image/games105/截图_20251112092021.png)
![alt text](/image/games105/截图_20251112092056.png)

允许任意三种基本旋转的组合 • 不包括绕同一轴旋转两次的情况 • XYZ、XZY、YZX、YXZ、ZYX、ZXY、XYX、XZX、YXY、YZY、ZXZ、ZYZ
绕物体旋转或绕世界旋转
![alt text](/image/games105/截图_20251112092208.png)
万象死锁：当两个局部轴被驱动到平行配置时，一个自由度会被“锁定”
𝑅𝑥(𝛼)𝑅𝑦(𝛽)𝑅𝑧(𝛾)
 3 parameters:  𝛼,𝛽,𝛾
 12 variations: XYZ, XZY, YZX, YXZ, ZYX, ZXY,
 XYX, XZX, YXY, YZY, ZXZ, ZYZ
 Intrinsic/Extrinsic rotations 
 • 易于组合吗？ √ 但难以创建特定旋转
 • 易于应用吗？ √ 需要三次矩阵相乘
 • 易于插值吗？ √ 需要处理奇点
 • 万向节锁定   × 旋转速度不恒定

* 旋转向量 / 轴角
* 轴角 （𝒖，𝜃）：使用以下方式表示旋转 
    * 一个向量 𝒖：旋转轴 
    * 一个标量 𝜃：旋转角度
* 旋转向量：表示旋转为
    * 𝜽 =𝜃𝒖 
    * 显然：
$
\theta=\|\theta\|\quad u=\frac{\theta}{\|\theta\|}
$

![alt text](/image/games105/截图_20251112222926.png)
![alt text](/image/games105/截图_20251112223047.png)
![alt text](/image/games105/截图_20251112223458.png)
![alt text](/image/games105/截图_20251112223558.png)
• 容易组合？  √  但难以操作
• 容易应用？  ×  需要转换为矩阵
• 容易插值？  √  线性插值可用，但不完美
• 无万向节锁  √   

* 四元数
从复数可以代表旋转，推理得来的四元数
参考: [复数旋转](https://zhuanlan.zhihu.com/p/85321120)
![alt text](/image/games105/截图_20251112224110.png)
![alt text](/image/games105/截图_20251112224241.png)
![alt text](/image/games105/截图_20251112225949.png)
![alt text](/image/games105/截图_20251112230148.png)
![alt text](/image/games105/截图_20251112230658.png)
四元数的实部与虚部
![alt text](/image/games105/截图_20251112230717.png)
![alt text](/image/games105/截图_20251112231300.png)
![alt text](/image/games105/截图_20251112231550.png)
![alt text](/image/games105/截图_20251112231748.png)
![alt text](/image/games105/截图_20251112231902.png)
![alt text](/image/games105/截图_20251112232053.png)
![alt text](/image/games105/截图_20251112232252.png)
![alt text](/image/games105/截图_20251112232435.png)
![alt text](/image/games105/截图_20251112232635.png)
![alt text](/image/games105/截图_20251112232741.png)
![alt text](/image/games105/截图_20251112233013.png)
插值：
![alt text](/image/games105/截图_20251112233129.png)
![alt text](/image/games105/截图_20251112233204.png)
![alt text](/image/games105/截图_20251112233310.png)
![alt text](/image/games105/截图_20251112233331.png)
![alt text](/image/games105/截图_20251112233733.png)
![alt text](/image/games105/截图_20251112233755.png)

• 容易组合吗？ √  需要归一化，难以操作
• 容易应用吗？ √ 四元数乘法SLERP
• 容易插值吗？ √ 需要处理奇点
• 无万向锁     √

## 03 Character Kinematics:Forward and Inverse Kinematics
### 链条运动学
![alt text](/image/games105/截图_20251113213958.png)
计算旋转
![alt text](/image/games105/截图_20251113221653.png)
计算相对旋转
![alt text](/image/games105/截图_20251113221731.png)
![alt text](/image/games105/截图_20251113222232.png)
![alt text](/image/games105/截图_20251113222446.png)
计算绝对坐标位置
![alt text](/image/games105/截图_20251113222730.png)
计算相对坐标
![alt text](/image/games105/截图_20251113224910.png)
![alt text](/image/games105/截图_20251113225241.png)
前向动力学：
![alt text](/image/games105/截图_20251113225400.png)
正向运动学：已知所有关节的旋转 𝑅𝑖，求 𝒙₀ 在全局坐标系 𝒙 下的坐标：
对于从根部到末端执行器的𝑖:
$
\begin{aligned}
 & Q_i=Q_{i-1}R_i \\
 & p_{i+1}=p_i+Q_i\boldsymbol{l}_i
\end{aligned}
$
$
x=p_\mathrm{E}+Q_\mathrm{E}x_0
$
$x=x_0$
从末端执行器到根部的 𝑖
$
x=l_{i-1}+R_ix
$
对于从关节 𝑘+1 到末端执行器的 𝑖：
$
\begin{aligned}
 & Q_i^{\prime}=Q_{i-1}^{\prime}R_i//(Q_0^{\prime}=I) \\
 & p_{i+1}^{\prime}=p_i^{\prime}+Q_i^{\prime}\boldsymbol{l}_i
\end{aligned}
$
$
x=p_\mathrm{E}^{\prime}+Q_\mathrm{E}^{\prime}x_0
$
$
x=l_{i-1}+R_ix
$

### 正向运动学
![alt text](/image/games105/截图_20251113230510.png)
膝盖，肘部:铰链关节 旋转关节
臀部，肩膀:球窝关节
![alt text](/image/games105/截图_20251113231414.png)
自由度：定义机械系统配置或状态的独立参数数量
![alt text](/image/games105/截图_20251113231856.png)
![alt text](/image/games105/截图_20251113231940.png)
关节限制
![alt text](/image/games105/截图_20251113232052.png)
姿态参数
关节通常按每个关节先于其子关节的顺序排列
![alt text](/image/games105/截图_20251113232253.png)
问题1：如果我们知道所有关节 𝑄𝑖 的方向，如何计算关节旋转？
问题2：我们应该如何允许可伸展的骨骼？
• BVH 文件 • 最常用的动作数据文件格式之一 
• 可在 Blender、FBX Review、Motion Builder 等中查看 
• 基于文本，易于阅读和编辑

• 格式
• 层级结构：定义角色的 T 形站姿 
• 动作：根节点位置及各关节的欧拉角
![alt text](/image/games105/截图_20251113232511.png)
参考:[BVH格式](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html)

### 逆向动力学
对于一个可以用一组参数 𝜽 描述的系统，以及由该系统给出的性质 𝒙
𝒙 =𝑓(θ)
前向运动问题：
• 已知 𝜽，我们需要计算 𝒙 
• 由于 𝑓 是已知的，计算很容易，结果是唯一的 
• 𝜽 的自由度通常比 𝒙 的自由度大得多。我们无法轻易调整 𝜽 来达到特定的 𝒙 值。

逆向动力学：
• 给定𝒙，我们需要找到一组有效参数𝜽，使得𝒙 = 𝒇(𝜽)
• 通常需要求解一个困难的非线性方程，这个方程可能有多个解 
• 𝒙通常是有意义的，并且可以以直观的方式设置

已知末端执行器的位置 𝒙，计算关节旋转$𝑅_i$
![alt text](/image/games105/截图_20251114092022.png)
给定末端执行器的位置 𝒙，计算关节旋转参数$𝜽_i$
![alt text](/image/games105/截图_20251114092203.png)
![alt text](/image/games105/截图_20251114092258.png)

两个关节点的ik：通过两个节点画圆，求交点
![alt text](/image/games105/截图_20251114092411.png)

一个解两关节点ik问题的简单方法
1. 旋转关节1，使得
$\|l_{0x}\|=\|l_{02}\|$
2. 旋转关节0，使得
$l_{0x}=l_{02}$
3. 如果必要，绕 $𝒍_0𝑥$ 旋转关节 0
![alt text](/image/games105/截图_20251114181456.png)
![alt text](/image/games105/截图_20251114181520.png)
![alt text](/image/games105/截图_20251114181550.png)

![alt text](/image/games105/截图_20251114204947.png)
函数的含义：
![alt text](/image/games105/截图_20251114205202.png)
![alt text](/image/games105/截图_20251114205221.png)

示意图，并非真实图像，每个虚线为等值面：
![alt text](/image/games105/截图_20251114205518.png)
• 找到一个有前景的方向来更新参数
• 沿着该方向移动参数适当的距离
• 重复这一过程，直到达到最优参数（或在几次迭代后停止）
![alt text](/image/games105/截图_20251114211007.png)

**循环坐标下降（CCD）**
沿坐标系的每个轴更新参数
循环遍历所有轴
![alt text](/image/games105/截图_20251114211136.png)
![alt text](/image/games105/截图_20251114211537.png)
迭代旋转每个关节，使末端执行器与关节和目标之间的向量对齐
易于实现，非常快速
“第一个”关节的运动比其他关节更多，可能需要多次迭代才能收敛，结果可能对初始解敏感

**梯度下降**
根据目标函数梯度的方向更新参数
梯度下降最快的方向
![alt text](/image/games105/截图_20251114212229.png)
往梯度反方向移动，函数变小，步长即学习率，
太大的学习率会导致错过极值点
![alt text](/image/games105/截图_20251115221904.png)

![alt text](/image/games105/截图_20251116121226.png)
补充：
函数推导公式
首先，将L2范数平方展开为向量内积的形式：
$
F(\theta) = \frac{1}{2} (f(\theta) - \tilde{x})^T (f(\theta) - \tilde{x})
$
我们使用微积分中的矩阵微分规则。设残差向量为 $r(\theta)=f(\theta)-\tilde{x}$ ,则：
$F(\theta)=\frac{1}{2}r(\theta)^Tr(\theta)$
对F(θ) 进行微分（使用$d(A^TB)=(dA)^TB+A^T(dB)$ ）：
$dF=\frac{1}{2}\left[(dr)^Tr+r^T(dr)\right]$
由于$(dr)^Tr$是一个标量（1x1矩阵），它等于其转置$r^T(dr)$ 。
因此：$dF=\frac{1}{2}\left[r^T(dr)+r^T(dr)\right]=r^T(dr)$
将$r(\theta)=f(\theta)-\tilde{x}$代入，所以$dr=d(f(\theta))$
函数 f(θ) 的微分可以用其雅可比矩阵 J表示：$df=Jd\theta$
其中雅可比矩阵 J的定义是
$J=\frac{\partial f}{\partial\theta}=
\begin{bmatrix}
\frac{\partial f_1}{\partial\theta_1} & \cdots & \frac{\partial f_1}{\partial\theta_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial\theta_1} & \cdots & \frac{\partial f_m}{\partial\theta_n}
\end{bmatrix}$
这里 f是 m 维向量函数，θ 是 n 维参数向量。
因此：
$dr=df=Jd\theta$
$\text{将 }dr=Jd\theta\text{ 代入 }dF=r^T(dr):$
$dF=r^T(Jd\theta)$
$\text{利用矩阵转置的性质 }(AB)^T=B^TA^T\text{,我们可以将 }r^TJ\text{ 视为 }(J^Tr)^T.$:$dF=(J^Tr)^Td\theta$
$\text{根据微分的定义,如果 }dF=g(\theta)^Td\theta\text{,那么梯度 }\nabla_\theta F=g(\theta)$
因此:
$\nabla_\theta F(\theta)=J^Tr$

**雅可比矩阵**:
![alt text](/image/games105/截图_20251116180840.png)
![alt text](/image/games105/截图_20251116181022.png)


计算f（θ）是哦那个任意机器学习框架： pytorch，tensorflow
使用其自动求导功能计算梯度
![alt text](/image/games105/截图_20251116182419.png)
铰链关节，通过推导可得其雅可比矩阵为：
![alt text](/image/games105/截图_20251116182655.png)
![alt text](/image/games105/截图_20251116182901.png)
球关节可以把其想象成按照欧拉角的三个方向分别进行旋转得到的最后的旋转，对应雅可比矩阵分别对x，y，z求导
![alt text](/image/games105/截图_20251116183651.png)
![alt text](/image/games105/截图_20251116183716.png)
先旋转x轴，再在基础上旋转y轴，在是z轴
![alt text](/image/games105/截图_20251116220140.png)
一阶方法，收敛可能较慢，需要在每次迭代时重新计算雅可比矩阵
$
\begin{aligned}
\boldsymbol{\theta}^{i+1} & =\boldsymbol{\theta}^i-\alpha\nabla_{\boldsymbol{\theta}}F(\boldsymbol{\theta}^i) \\
 & =\theta^i-\alpha J^T\Delta
\end{aligned}
$
一阶最优条件：𝜽∗ 是局部最小值
$\nabla_\theta F(\boldsymbol{\theta}^*)=0$
![alt text](/image/games105/截图_20251116224044.png)
**二次规划问题**
$
\min_{\theta}F(\theta)=\frac{1}{2}\boldsymbol{\theta}^TA\boldsymbol{\theta}+\boldsymbol{b}^T\boldsymbol{\theta}
$
其中 𝐴 是正定的：
𝐴 =𝐴𝑇,𝜽𝑇𝐴𝜽 ≥ 0forany𝜽
![alt text](/image/games105/截图_20251117205802.png)
**高斯-牛顿方法**
![alt text](/image/games105/截图_20251117210042.png)
![alt text](/image/games105/截图_20251117210104.png)
![alt text](/image/games105/截图_20251117212436.png)
![alt text](/image/games105/截图_20251117212452.png)
![alt text](/image/games105/截图_20251118221841.png)
![alt text](/image/games105/截图_20251118222000.png)
![alt text](/image/games105/截图_20251118225250.png)
![alt text](/image/games105/截图_20251118225341.png)
只有铰关节，并且多个对应点时，可逆。
![alt text](/image/games105/截图_20251118225739.png)
矮胖矩阵对应欠压制问题，高瘦矩阵对应约束过多，以至无法达到对应点。
$\theta=\theta^0-\alpha J^+\Delta$
一般情况下只有在函数附近才是有效的，因此会增加一个学习率
通常比梯度下降法和雅可比矩阵法要快
![alt text](/image/games105/截图_20251118225922.png)
𝐽𝐽𝑇 / 𝐽𝑇𝐽有时候不可逆，或接近不可逆，求解值可能过大   ，带来不稳定的效果，因此会在其中加入一个阻尼项，能够保证（）内部矩阵可逆。
也称为列文伯格-马夸特算法
![alt text](/image/games105/截图_20251118231956.png)
其含义为目标函数后加了一个约束项，为了移动到目标点，每个点都会发生旋转，而约束项则保证用每个关节点用尽量少旋转完成移动。 
![alt text](/image/games105/截图_20251118232035.png)
针对关节旋转使用不同的权重。
![alt text](/image/games105/截图_20251118232434.png)
多个优化函数结合形成整个姿态，根节点可以选择不同的位置
![alt text](/image/games105/截图_20251118232658.png)

### 总结
![alt text](/image/games105/截图_20251118233013.png)
![alt text](/image/games105/截图_20251118233101.png)


## 作业1
### 安装anaconda
下载minianaconda
具体安装方法按照官网

修改环境变量
安装路径\Miniconda3 

安装路径\Miniconda3\Scripts

安装路径\Miniconda3\Library\bin 
![alt text](/image/games105/截图_20251119214524.png)

创建虚拟环境，安装包等
conda create -n games105 python=3.8
conda activate games105
conda install numpy scipy
pip install panda3d

### 前向运动学
#### 作业1
#### 作业2
#### 作业3
## 逆向运动学
#### 作业1
#### 作业2
#### 作业3

## 04 Character Kinematics (cont.)  & Keyframe Animation
### 重定向
Tpose and Apose
不同pose导致旋转同样角度后最终动作不同
先转换到tpose再转换到最终动作
![alt text](/image/games105/截图_20251130222116.png)
![alt text](/image/games105/截图_20251130224653.png)

锁链的重定向
![alt text](/image/games105/截图_20251130225115.png)
![alt text](/image/games105/截图_20251130225443.png)
![alt text](/image/games105/截图_20251130230047.png)

同理可推出人物骨骼
![alt text](/image/games105/截图_20251130230325.png)

### ik and fk 复习
当有轴只可以伸长时，其他旋转方法相同
![alt text](/image/games105/截图_20251201210657.png)

多个ik目标
![alt text](/image/games105/截图_20251201211830.png)

### 关键帧动画
根据一系列的点生成对应的函数，再进行插值或外推
![alt text](/image/games105/截图_20251201212801.png)
![alt text](/image/games105/截图_20251201212825.png)

**Stepped Interpolation**
![alt text](/image/games105/截图_20251201213118.png)
![alt text](/image/games105/截图_20251201213152.png)

**Linear Interpolation**
![alt text](/image/games105/截图_20251201213327.png)

平滑方法：
一阶导数相同、二阶导数相同
![alt text](/image/games105/截图_20251201213452.png)

**Polynomial Interpolation多项式插值**
![alt text](/image/games105/截图_20251201213942.png)
![alt text](/image/games105/截图_20251201214029.png)
![alt text](/image/games105/截图_20251201214121.png)

在边缘处产生较大震荡，
所以低阶多项式更受欢迎
![alt text](/image/games105/截图_20251201214453.png)

**Spline Interpolation**
使用低次分段多项式进行插值
![alt text](/image/games105/截图_20251201214613.png)

使用低阶分段多项式进行插值 
• 一阶 → 分段线性插值
![alt text](/image/games105/截图_20251201214807.png)

• 三次多项式 → 三次样条
![alt text](/image/games105/截图_20251201215036.png)

**三次样条**
有n段，所以有4n个未知数
![alt text](/image/games105/截图_20251201215301.png)
![alt text](/image/games105/截图_20251201215507.png)
缺点：
没有局部控制，每次更改一个点，将会影响整个曲线
计算复杂，当n较大时需要处理庞大的线性系统

**Cubic Hermite Splines**
![alt text](/image/games105/截图_20251201221921.png)
只看x1-x2这一段，因此x =x1时候，t = 0，x =x2时，t = 1
![alt text](/image/games105/截图_20251201222859.png)
![alt text](/image/games105/截图_20251201224618.png)
![alt text](/image/games105/截图_20251202205915.png)
![alt text](/image/games105/截图_20251202210122.png)
![alt text](/image/games105/截图_20251202212509.png)
当y1，y2点的位置，m1，m2为速度方向时：
![alt text](/image/games105/截图_20251202212626.png)
Example: Curve Tool of PowerPoint
Hermite spline的特例Catmull-Rom Spline
即对m1，m2定义
![alt text](/image/games105/截图_20251202213114.png)

**Interpolation of Rotations**
四种表示方法：
![alt text](/image/games105/截图_20251202213436.png)
![alt text](/image/games105/截图_20251202213530.png)

**四元数的球面线性插值（SLERP）**
使用slerp对四元数进行插值（线性）
![alt text](/image/games105/截图_20251202213738.png)
[非线性插值](https://splines.readthedocs.io/en/latest/rotation/kochanek-bartels.html)
Ken Shoemake. 1985 
Animating rotation with quaternion curves. SIGGRAPH Computer Graphics

## 05 Data-driven Character Animation
### Motion Capture
数字化记录人类、动画和物体的动作
娱乐：游戏、电影、虚拟偶像、元宇宙
运动：专业训练，表现优化
医学：骨科、损伤诊断与治疗
机器人技术：跟踪与定位

**动作捕捉的历史**
• 利用摄影研究运动
• 跟踪抠像

**现代动作捕捉系统**
机械动作捕捉
• Exoskeleton外骨骼
惯性动作捕捉
• Inertial Measurement Unit (IMU)惯性测量单元（IMU）
• Accelerometers (3dof) + axis gyroscope (3dof)加速度计（3自由度）   轴向陀螺仪（3自由度）
• 可选择其他传感器
光学动作捕捉
• Reflective/light-emitting markers• 反光/发光标记
• Multi-view geometry• 多视角几何 
• Solve body motions based on marker positions• 基于标记位置求解身体动作
多摄像机无标记动作捕捉
使用深度摄像头的无标记动作捕捉
单目视频运动估计
使用稀疏传感器的运动估计:vr设备

### Motion Retargeting
Retarget a motion to drive a character with 
• Different number of bones 
• Different bone names
• Different reference pose
• Different bone ratios
• Different skeletal structure 
可能的重新定向流程 
• 映射骨骼名称 
• 缩放平移 
• 复制或重新定向关节旋转以修复参考姿势 
• 使用IK进行后处理 
• 脚滑动 
• 自身穿透

### Motion Transition
前后两段动作取若干帧采用线性插值：
![alt text](/image/games105/截图_20251202223301.png)
前后两段动作取若干帧采用其他函数插值：
![alt text](/image/games105/截图_20251202224018.png)
主要采用后一段或前一段取出若干帧进行插值

**Facing Frame**
一种特殊的坐标系统，水平随角色移动，其中一个轴指向角色的“朝向方向”
![alt text](/image/games105/截图_20251202224511.png)
R的可能定义：
𝑅 是将全局坐标系的 z 轴与航向方向对齐的 y 轴旋转
𝑅 是沿 y 轴旋转，使全局坐标系的 x 轴与肩膀和臀部之间向量的平均方向对齐
分解根旋转为 𝑅0 = 𝑅𝑦𝑅𝑥z
分解旋转：
先沿着面上的一个轴进行旋转，再沿着垂直于当前面的轴线进行旋转
![alt text](/image/games105/截图_20251202225144.png)
![alt text](/image/games105/截图_20251204214338.png)
![alt text](/image/games105/截图_20251204214410.png)

**Motion Transition**
将后一段的坐标系转换到前一段上：
![alt text](/image/games105/截图_20251204220907.png)
或者使用不带roottranslation的动作数据，让人物沿着路径移动

### Motion Graphs
根据以下内容计算生成动作：
• 用户控制 
• 同一环境中的物体 
• 其他角色的动作

**分段运动数据**
• 距离图（根据需求确定，可以是动作旋转，距离等）
• 每个像素表示一对姿势之间的差异 
• 局部最小值是潜在的转换点
![alt text](/image/games105/截图_20251204223249.png)

**动作合成**
• 状态机 
    • 节点表示动作片段 
    • 边表示潜在的转场 
    • 当需要时触发转场 
        • 用户输入 
        • 剧集结束 
    • 检查下一片段的直接连接 
        • 可能需要更深入的搜索
![alt text](/image/games105/截图_20251204223436.png)
根据用户输入、环境等更新动作，（后续可加入ik等）
![alt text](/image/games105/截图_20251204223705.png)

**Motion Matching**
切换更加自然，（因为每帧进行计算最邻近动作）
制作相对简单，但是控制性较差
根据用户反馈等，以及动作数据集生成距离函数，选择距离最近的动作，并进行插值。
![alt text](/image/games105/截图_20251204224618.png)

## 06 Learning-based Character Animation
### 交互式角色动画
**Motion Graphs**
**Motion Fields / Motion Matching**
![alt text](/image/games105/截图_20251209222600.png)
![alt text](/image/games105/截图_20251209222653.png)
![alt text](/image/games105/截图_20251209222743.png)
相比motionfields，MotionMatching使用最近状态，根据用户输入，并且通过混合的方式平滑。
![alt text](/image/games105/截图_20251209222805.png)
如何定义姿态的距离函数，可能的特征向量
![alt text](/image/games105/截图_20251209223635.png)
![alt text](/image/games105/截图_20251209223822.png)
### 人体运动的统计模型
**人体动作的低维性**
• 协调的手臂/腿部运动 
• 肌肉骨骼结构 
• 物理法则
![alt text](/image/games105/截图_20251209224112.png)
**主成分分析 (PCA)**
• 一种用于发现各维度之间的相关性降维的技术
![alt text](/image/games105/截图_20251209225821.png)
![alt text](/image/games105/截图_20251209225953.png)
![alt text](/image/games105/截图_20251209230256.png)
可以证明，𝒖 是 𝑋𝑇𝑋 的特征向量，对应于最大的特征值
![alt text](/image/games105/截图_20251211221614.png)
当多次取u时，每次都在上一次降维基础上，取新的轴，继续取最大方差
![alt text](/image/games105/截图_20251211225405.png)
因此有n个方差，把方差从大到小排列，即对应的n个特征值，其中的特征向量u互相垂直。
![alt text](/image/games105/截图_20251211230357.png)
保留多个主成分
![alt text](/image/games105/截图_20251213175255.png)
![alt text](/image/games105/截图_20251213175232.png)
动作方差越小越自然，
![alt text](/image/games105/截图_20251213175446.png)
![alt text](/image/games105/截图_20251213175947.png)
除了保持ik到达目标点距离最小值外，保持动作最自然。
![alt text](/image/games105/截图_20251213180257.png)
**数据分布**
假设数据分布按照正态分布的形式，根据平均值与方差带入后可得：
![alt text](/image/games105/截图_20251214114918.png)
![alt text](/image/games105/截图_20251214115047.png)
![alt text](/image/games105/截图_20251214115409.png)
推导过程：
![alt text](/image/games105/截图_20251214122851.png)
![alt text](/image/games105/截图_20251214122948.png)
![alt text](/image/games105/截图_20251214123033.png)
![alt text](/image/games105/截图_20251214123130.png)
因此之前的加方差可以带换成
![alt text](/image/games105/截图_20251214123232.png)
![alt text](/image/games105/截图_20251214123427.png)
![alt text](/image/games105/截图_20251214123539.png)
带运动先验的运动合成：
给定从一组数据点 D 学习到的运动先验 𝑝(𝒙)，合成一个运动 𝒙 以最小化目标函数
注意：𝑥 可以表示一个姿势 𝜽 或一个动作片段 → 一个姿势序列 {𝜽𝒕} 或动作的任何特征 → 例如 PCA 中的 𝑤𝑘
![alt text](/image/games105/截图_20251214123800.png)
![alt text](/image/games105/截图_20251214124326.png)
其他分布模型：
![alt text](/image/games105/截图_20251214124551.png)
![alt text](/image/games105/截图_20251214124628.png)

**运动模型**
关于动作序列的两种视角：
![alt text](/image/games105/截图_20251215081127.png)
z为控制参数、潜变量、当前角色的状态（老人小孩等）
![alt text](/image/games105/截图_20251215081251.png)
![alt text](/image/games105/截图_20251215081607.png)
给出控制条件，生成一段动作
![alt text](/image/games105/截图_20251215082153.png)
与前面一系列动作相关：
![alt text](/image/games105/截图_20251215082413.png)
只与前面一帧动作相关：
![alt text](/image/games105/截图_20251215082534.png)
自回归模型Autoregressive Model
因此只需要前一帧到目前帧的函数变换
![alt text](/image/games105/截图_20251215082704.png)
![alt text](/image/games105/截图_20251215082859.png)
因此使用插值的算法可得到当前帧的位置
![alt text](/image/games105/截图_20251215083312.png)
神经网络
w为权重，b为偏移量，xi为加权值，σ为极大函数，当加权到某一最大值后输出。
![alt text](/image/games105/截图_20251215083405.png)
多层神经网络
![alt text](/image/games105/截图_20251215083946.png)
![alt text](/image/games105/截图_20251215084009.png)
![alt text](/image/games105/截图_20251215084031.png)
![alt text](/image/games105/截图_20251215084140.png)
找到w1，b1，w2，b2...使得函数最小：
![alt text](/image/games105/截图_20251215084239.png)
![alt text](/image/games105/截图_20251215084842.png)
随机梯度下降，循环更新θ，从数据集里选取小的子集，算出近似梯度，沿负梯度方向乘步长进行更新
![alt text](/image/games105/截图_20251215085613.png)

**歧义问题**
可能向各个方向走，因此平均后可能方向朝前，产生错误。因此增加控制量z
![alt text](/image/games105/截图_20251215090214.png)
![alt text](/image/games105/截图_20251215090306.png)
**PFNN: Phase-Functioned Neural Networks相位函数神经网络**
![alt text](/image/games105/截图_20251215090528.png)
![alt text](/image/games105/截图_20251215090742.png)
由每个专家负责一部分，将最终结果按照权重进行混合
![alt text](/image/games105/截图_20251215090810.png)
将所有专家合并为一个专家（即参数按照一定的权重组合成一个函数）
![alt text](/image/games105/截图_20251215090910.png)
![alt text](/image/games105/截图_20251215091151.png)
![alt text](/image/games105/截图_20251215091214.png)
![alt text](/image/games105/截图_20251215091230.png)
当不再是人物动作，如狗等动物时，如何定义相位以及相位权重?通常使用门网络来计算wi
![alt text](/image/games105/截图_20251215091610.png)
门网络输入参数，增加角速度的考虑，在狗的动作上有较好的表现，
对两个脚分布定义一个相位函数，以及手部分别定义相位函数。
通过傅里叶变换学到一个相位函数。
![alt text](/image/games105/截图_20251215091740.png)

**Generative Models生成模型**
动作的概率分布函数是个复杂的函数，训练f（z）函数将正态分布函数变形成概率分布函数
![alt text](/image/games105/截图_20251215214801.png)
常见的方法：
Variational Autoencoders变分自编码器:编码器f(x),解码器g(x)
Generative Adversarial Network生成对抗网络：一个编码器，一个判别器，判断是否在概率分布函数上，当无法判断是否在其上时，则f（z）能够实现变换
![alt text](/image/games105/截图_20251215215017.png)
Normalizing Flows归一化流：类似Variational Autoencoders只不过使用了逆运算作为解码器
扩散模型
![alt text](/image/games105/截图_20251215220800.png)