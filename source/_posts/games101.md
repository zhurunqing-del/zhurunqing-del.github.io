---
title: games101
date: 2025-07-26 16:55:25
tags: ["games", "图形学"]
katex: true
---
games101笔记，以及作业。之前看过一编，现在有时间了过来整理一下笔记以及重新做一下相关的作业。

## 01 overview

课程内容主要包含

* **光栅化**（rasterization）：把三维空间的几何形体显示在屏幕上。实时（30fps）是一个重要的挑战。

* **几何表示**（geometry）：如何表示曲线、曲面、拓扑结构等

* **光线追踪**（ray tracing）：慢但是真实。实时是一个重要的挑战。

* **动画/模拟**（animation/simulation）：譬如扔一个球到地上，球如何反弹、挤压、变形等

## 02 Review of Linear Algebra
### 向量（Vector）
向量加法：使用三角形法则或者平行四边形法则
向量点乘（Dot）
向量叉乘（Cross）

### 矩阵（Matrices）
矩阵运算：加减乘法以及各个运算律
向量乘法转换为矩阵乘法

## 03-04 Transformation
**二维变换（2D transformation**

旋转矩阵（Rotation）:  $\mathbf{R}(\theta) = \begin{bmatrix}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{bmatrix}$

缩放矩阵（Scale）：$\mathbf{S}(s_x,s_y)= \begin{bmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & 1
\end{bmatrix}$
切变矩阵（Shear）：$\mathbf{H}(k) = \begin{bmatrix}
1 & k & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$

平移矩阵（Translation）：$\mathbf{T}(t_x,t_y)= \begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}$

以上的矩阵均为齐次坐标矩阵。前三种被称为线性变换，加入平移变换后统称为仿射变换。变换的次序非常重要，交换非同类的矩阵的乘法次序往往得不到相同结果

**齐次坐标（Homogeneous coordinates**

统一平移矩阵和线性变换矩阵，使其可以直接参与运算
为向量拓展一个维度，使得一个三维变量可以区分表示为向量和点（w为1为点，w为0为向量，保证向量的平移不变性）

**三维变换（3D Transformation）**
旋转矩阵（Rotation）：
$\mathbf{R}_x = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$
$\mathbf{R}_y = \begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$
$\mathbf{R}_z = \begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$
缩放矩阵（Scale）：$\mathbf{S} = \begin{bmatrix}
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$
平移矩阵（Translation）：$\mathbf{T} = \begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}$

**万向节死锁**
由于欧拉角（Euler）表示旋转时存在旋转轴的次序，当中间旋转轴旋转使得前后两个轴对齐时，就失去了一个旋转自由度，出现万向节死锁。可以使用四元数（Quaternion）解决该问题。

**MVP矩阵变换（Model-View-Projection Matrices）**
* 模型矩阵（Model）：对模型上的点应用模型的平移、旋转和缩放变换，将本地坐标转换到世界坐标
* 视图矩阵（View）：对世界坐标应用摄像机平移、旋转的逆变换，将世界坐标转换为视图坐标
* 投影矩阵（Projection）：根据投影方式应用投影变换，将视图坐标转换为裁剪空间坐标
通俗理解：

* Model为调整模型位置和Pose
* View为调整摄像机位置
* Projection为拍照（3D空间转2D屏幕）

法线的模型矩阵：

* 当Model变换中存在非等比缩放/切变变换等变换时，模型的法线如果使用和顶点一样的Model矩阵，那么法线就会出现异常。
* 此时需要对法线应用单独的Model矩阵，和原来Model矩阵的关系为左上3x3部分互为逆矩阵

**投影变换（Projection）**
正交投影（Orthographic Projection）

将任意的立方体通过平移和缩放，转换为标准化（Canonical）立方体（ 坐标范围为$[-1 , 1]^3$ )
$\mathbf{P}_{ortho} = \begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & - \frac{r+l}{r-l} \\
0 & \frac{2}{t-b} & 0 & - \frac{t+b}{t-b}\\
0 & 0 & \frac{2}{n-f} & - \frac{n+f}{n-f}\\
0 & 0 & 0 & 1
\end{bmatrix}$
(right、left、top、bottom、near、far六个平面)


透视投影（Perspective Projection）
$\mathbf{M}_{ortho->persp} = \begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0\\
0 & 0 & n+f & -nf \\
0 & 0 & 1 & 0
\end{bmatrix}$

$\mathbf{P}_{persp} =\mathbf{P}_{ortho}* \mathbf{M}_{ortho->persp}
=\begin{bmatrix}
\frac{2n}{r-l} & 0 & - \frac{r+l}{r-l} & 0 \\
0 & \frac{2n}{t-b} & - \frac{t+b}{t-b} & 0\\
0 & 0 & \frac{n+f}{n-f} & - \frac{2nf}{n-f}\\
0 & 0 & 1 & 0
\end{bmatrix}$

将视锥体（Frustum）通过矩阵变换，转换为Canonical立方体
参数：fovy垂直视场角（Field of View），aspect宽高比，n、f、r、l、t、b为六个边界平面
具体可参考作业1：

其他知识：

裁剪空间坐标（Clip Space）和NDC标准化设备坐标（Normalized Device Coordinate）：理论上投影变换只是将视图坐标转换为裁剪空间坐标 
 ，需要后续进行透视除法（同除以w）才能转换为NDC 
进行“压缩”变换时一般点的坐标变换情况：x,y的值向0靠近，z值变小/远

## Lecture 05~06 Rasterization 光栅化

**光栅化基础知识**
光栅化（Rasterization）：在屏幕上进行绘图
屏幕（Screen）：像素的二维数组
像素（Pixel）：同时只能显示一种颜色的最小显示单位（锯齿产生的根本原因）
屏幕坐标：以左上角或者左下角为原点的二维坐标
视口变换：将NDC映射到屏幕上得到屏幕坐标

**三角形网格（Triangle Meshes）**
三角形是最基础的多边形
三角形的重心插值计算简单
三个点唯一确定一个平面
三角形一定是凸多边形，内外定义清晰易于求解

**采样（Sampling）**
最简单的采样方法：

对屏幕中每一个像素中心，判断其和三角形的内外关系，进行01的颜色赋值（使用叉乘判断内外关系）
边界情况自定义处理规则（比如Top-Left规则）
加速采样：只采样三角形在屏幕中的轴对齐包围盒AABB（Axis-Aligned Bounding Box）内部的像素，缩小采样范围

采样带来的问题：

简单的采样由于使用了根据像素中心的位置进行01的颜色填充，会产生严重的锯齿问题
对高分辨率的图像进行低分辨率的采样，会因为信息丢失产生摩尔纹（Moire）
车轮倒转，时间上的采样带来的问题

**频域（Frequency Domain）和时域（Time Domain）解释锯齿/走样**
频域和时域

频域：描述信号在频率方面特性的一种坐标系
时域：描述数学函数或物理信号对时间关系的一种坐标系（图像信号的空间分布也可视为一种时间关系）
时域的卷积等于频域的乘积，反之亦然
傅里叶级数展开与傅里叶变换

傅里叶级数展开：任意周期函数都可以写成一系列正弦和余弦函数的线性组合以及一个常数项


傅里叶变换：正变换会将时域图转换为频域图，逆变换会将频域图转换为时域图
图片本身可以视为一张傅里叶时域图，转换为频域图时默认认为图片边缘进行了平铺重复的拼接
滤波（Filtering）：和卷积/加权平均等价
$f(x) = \frac{\mathbf{A}}{2}+\frac{2 \mathbf{A} \cos( t\pi )}{\pi}  - \frac{2 \mathbf{A} \cos( 3t\pi )}{3\pi} + \frac{2 \mathbf{A} \cos( 5t\pi )}{5\pi} -\frac{2 \mathbf{A} \cos( 7t\pi )}{7\pi} $

低通滤波：低频信号通过，实现模糊效果
高通滤波：高频信号通过，实现边缘检测/图像锐化效果
带通滤波：指定范围频率信号通过
采样率不够导致频谱混叠是走样产生的根本原因

**抗锯齿（AntiAliasing）**

基于超采样的抗锯齿

* SSAA 超采样抗锯齿（Super-Sampling AA）
严格超采样（比如采用2x2的4倍超采），然后降分辨率到目标分辨率
质量极高，开销极大，实时渲染中通常不会使用
* MSAA 多重采样抗锯齿（Multi-Sampling AA）
对SSAA的近似和改进
只在几何边缘（多边形边界）进行超采样。即对同一个几何体，只进行一次采样
存在问题：
只对几何边缘的锯齿有效，对纹理锯齿、阴影锯齿等颜色边界不产生效果。
延迟渲染管线（Deffered Shading）中无法使用MSAA
* TAA 时间抗锯齿（Temporal AA）
复用历史帧的信息进行混合，达成“超采样”
存在问题：对于运动物体会产生拖影（Ghosting）现象

基于后处理的抗锯齿

* FXAA 快速近似抗锯齿（Fast Approximate AA）
基于亮度差异检测边缘，然后进行模糊处理
优点：开销很低，兼容性好（对几何、纹理、阴影锯齿均有效）
缺点：可能模糊非锯齿的边界信息（比如UI，文字）
* MLAA 形态抗锯齿（Morphological AA）
形态学分析边缘几何模式，然后进行处理
更智能的边缘处理，开销较大
* SMAA 亚像素形态抗锯齿（Subpixel Morphological AA）
基于MLAA的优化改进，支持亚像素分析
优点：可处理亚像素精度，画质最佳且可以保留高频信息
缺点：实现复杂，需要结合TAA解决时间性锯齿

基于深度学习的抗锯齿

* DLSS 深度学习超采样（Deep Learning Super Sampling）：NVIDIA的超采样技术
* FSR 保真度超分辨率（FidelityFX Super Resolution）：AMD的超采样技术

## Assignment1

![项目截图](/image/games101/assignment1.png)


``` bash
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    float angle_rad = rotation_angle * EIGEN_PI / 180.0f;
    Eigen::Matrix4f Rotate;
    Rotate<< std::cos(angle_rad), -std::sin(angle_rad), 0,0,
        std::sin(angle_rad), std::cos(angle_rad), 0,0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    return Rotate *model;
}

//提高绕任意穿过圆心的轴旋转
Eigen::Matrix4f get_model_matrix(Eigen::Vector3f axis, float angle)
{
    // 1. 确保轴为单位向量
    axis.normalize();

    // 2. 使用Eigen的AngleAxis创建旋转
    Eigen::AngleAxisf rotation_vector(angle, axis);

    // 3. 将AngleAxis转换为3x3旋转矩阵
    Eigen::Matrix3f rotation_matrix = rotation_vector.toRotationMatrix();

    // 4. 构造4x4齐次变换矩阵
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    model.block<3, 3>(0, 0) = rotation_matrix; // 左上角3x3赋值为旋转矩阵

    // 平移分量保持为0，坐标原点为旋转中心

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

   

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float fov_rad = eye_fov * EIGEN_PI / 180.0f;

    
    float n = zNear; float f = zFar;
    float t = -zNear * std::tan(fov_rad / 2);
    float b = -t;
    float r = aspect_ratio * t;
    float l = -r;

    Eigen::Matrix4f Mscale;

    Mscale << 2 /( r - l), 0, 0, 0,
        0, 2 /( t - b), 0, 0,
        0, 0, 2 / (n - f), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f Mtranslate;
    Mtranslate << 1, 0, 0, -(r + l) / ( r - l),
        0, 1, 0, -(t + b) / (t-b),
        0, 0, 1,-(n + f)  / (n-f),
        0, 0, 0, 1;

    Eigen::Matrix4f Mpersp;
    Mpersp << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;

    return Mscale *Mtranslate * Mpersp *projection;
}

```
作业参考: [透视投影矩阵参考](https://zhuanlan.zhihu.com/p/122411512)




