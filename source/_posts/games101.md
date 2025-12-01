---
title: games101
date: 2025-07-26 16:55:25
tags: ["games", "图形学"]
katex: true
---
games101笔记以及作业。之前看过一遍，现在有时间了过来整理一下笔记以及重新做一下相关的作业。

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
### 二维变换（2D transformation

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

### 齐次坐标（Homogeneous coordinates

统一平移矩阵和线性变换矩阵，使其可以直接参与运算
为向量拓展一个维度，使得一个三维变量可以区分表示为向量和点（w为1为点，w为0为向量，保证向量的平移不变性）

### 三维变换（3D Transformation）
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

### 万向节死锁
由于欧拉角（Euler）表示旋转时存在旋转轴的次序，当中间旋转轴旋转使得前后两个轴对齐时，就失去了一个旋转自由度，出现万向节死锁。可以使用四元数（Quaternion）解决该问题。

### MVP矩阵变换（Model-View-Projection Matrices）
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

### 投影变换（Projection）
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


## Lecture 05~06 Rasterization 光栅化

### 光栅化基础知识
光栅化（Rasterization）：在屏幕上进行绘图
屏幕（Screen）：像素的二维数组
像素（Pixel）：同时只能显示一种颜色的最小显示单位（锯齿产生的根本原因）
屏幕坐标：以左上角或者左下角为原点的二维坐标
视口变换：将NDC映射到屏幕上得到屏幕坐标

### 三角形网格（Triangle Meshes）
三角形是最基础的多边形
三角形的重心插值计算简单
三个点唯一确定一个平面
三角形一定是凸多边形，内外定义清晰易于求解

### 采样（Sampling）
最简单的采样方法：

对屏幕中每一个像素中心，判断其和三角形的内外关系，进行01的颜色赋值（使用叉乘判断内外关系）
边界情况自定义处理规则（比如Top-Left规则）
加速采样：只采样三角形在屏幕中的轴对齐包围盒AABB（Axis-Aligned Bounding Box）内部的像素，缩小采样范围

采样带来的问题：

简单的采样由于使用了根据像素中心的位置进行01的颜色填充，会产生严重的锯齿问题
对高分辨率的图像进行低分辨率的采样，会因为信息丢失产生摩尔纹（Moire）
车轮倒转，时间上的采样带来的问题

### 频域（Frequency Domain）和时域（Time Domain）解释锯齿/走样
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

### 抗锯齿（AntiAliasing）

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

![msaa](/image/games101/05-061.png)
0、2、3号点位于同一几何体内，优化为一次采样


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

深度缓冲（Depth Buffer/Z-Buffer）

* 解决物体遮挡的方案
* 画家算法（Painter's Algorithm）
对物体进行排序，从后往前进行覆盖渲染
问题：排序本身比较复杂。可能出现循环遮挡现象，无法排序
* 渲染过程中按像素进行深度测试（Z-Test），未通过则不进行渲染，通过才进行渲染并更新深度缓冲Z-Buffer
* 透明物体渲染：只进行Z-Test，禁用深度写入Z-Write，然后根据测试结果和透明因子进行颜色混合


![深度缓冲](/image/games101/05-062.png)
复杂的遮挡关系，使得画家算法不再可用

## Assignment2


![项目截图](/image/games101/assignment21.png)

``` bash

static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector2f P = Vector2f(float(x), float(y));   // 待测点
    Eigen::Vector2f A = _v[0].head<2>();    // 三角形顶点A(x,y)
    Eigen::Vector2f B = _v[1].head<2>();
    Eigen::Vector2f C = _v[2].head<2>();

    // 计算向量
    Eigen::Vector2f AB = B - A;
    Eigen::Vector2f BC = C - B;
    Eigen::Vector2f CA = A - C;

    Eigen::Vector2f AP = P - A;
    Eigen::Vector2f BP = P - B;
    Eigen::Vector2f CP = P - C;

    float cross1 = cross2D(AB, AP);
    float cross2 = cross2D(BC, BP);
    float cross3 = cross2D(CA, CP);

    bool has_neg = (cross1 < 0) || (cross2 < 0) || (cross3 < 0);
    bool has_pos = (cross1 > 0) || (cross2 > 0) || (cross3 > 0);

    return !(has_neg && has_pos);

}

rst::BoundingBox rst::rasterizer::GetBounding(const std::array<Eigen::Vector4f, 3>& v) {

    rst::BoundingBox bbox;
    bbox.min = Eigen::Vector2f(v[0].x(), v[0].y());
    bbox.max = bbox.min;


    for (const auto& vec : v) {
        bbox.min.x() = std::min(bbox.min.x(), vec.x());
        bbox.min.y() = std::min(bbox.min.y(), vec.y());

        bbox.max.x() = std::max(bbox.max.x(), vec.x());
        bbox.max.y() = std::max(bbox.max.y(), vec.y());
    }

    return bbox;

}



//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    std::array<Eigen::Vector4f, 3> v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    rst::BoundingBox bbox = rst::rasterizer::GetBounding(v);
    
    // iterate through the pixel and find if the current pixel is inside the triangle
    int  xmin = static_cast<int>(std::floor(bbox.min[0])); 
    int  ymin = static_cast<int>(std::floor(bbox.min[1]));  
    int  xmax = static_cast<int>(std::floor(bbox.max[0]))+1;
    int  ymax = static_cast<int>(std::floor(bbox.max[1]))+1;

    for (int m = xmin; m < xmax; m++)
    {
        for (int n = xmin; n < xmax; n++)
        {
            if (insideTriangle(m+0.5,n+0.5,t.v))
            {
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(m, n, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(m, n);
                if (z_interpolated < depth_buf[index]) {
                    Eigen::Vector3f p;
                    p << m, n, z_interpolated;
                    set_pixel(p, t.getColor());
                    depth_buf[index] = z_interpolated;
                }
            }

        }
    }
    
}


```


提高(主要修改了以下部分)，采用了ssaa的方式：
![项目截图](/image/games101/assignment22.png)

``` bash

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{



  ```````````



    for (int m = 0; m < width; m++)
    {
        for (int n = 0; n < height; n++)
        {
            // 计算最终像素颜色和深度
            Eigen::Vector3f p;
            int index = get_index(m, n);
            float z_interpolated = (depth_buf_0[index] + depth_buf_1[index] + depth_buf_2[index] + depth_buf_3[index]) / 4.0f;
            p << m, n, z_interpolated;
            Eigen::Vector3f color = (frame_buf_0[index] + frame_buf_1[index] + frame_buf_2[index] + frame_buf_3[index]) / 4.0f;
            depth_buf[index] = z_interpolated;
            // 正确计算平均颜色
            set_pixel(p, color);
        }
    }
}






void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    std::array<Eigen::Vector4f, 3> v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    rst::BoundingBox bbox = rst::rasterizer::GetBounding(v);
    
    // iterate through the pixel and find if the current pixel is inside the triangle
    int  xmin = static_cast<int>(std::floor(bbox.min[0])); 
    int  ymin = static_cast<int>(std::floor(bbox.min[1]));  
    int  xmax = static_cast<int>(std::floor(bbox.max[0]))+1;
    int  ymax = static_cast<int>(std::floor(bbox.max[1]))+1;

    for (int m = xmin; m < xmax; m++)
    {
        for (int n = ymin; n < ymax; n++)
        {
            if (insideTriangle(m + 0.25, n + 0.25, t.v))
            {
                auto [alpha, beta, gamma] = computeBarycentric2D(m + 0.25, n + 0.25, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                int index = get_index(m, n);
                if (z_interpolated < depth_buf_0[index]) {
                    depth_buf_0[index] = z_interpolated;
                    frame_buf_0[index] = t.getColor();
                }
            }
            if (insideTriangle(m + 0.25, n + 0.75, t.v))
            {
                // If so, use the following code to get the interpolated z value.
                auto [alpha, beta, gamma] = computeBarycentric2D(m + 0.25, n + 0.75, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(m, n);
                if (z_interpolated < depth_buf_1[index]) {
                    depth_buf_1[index] = z_interpolated;
                    frame_buf_1[index] = t.getColor();
                }
            }
            if (insideTriangle(m + 0.75, n + 0.25, t.v))
            {
                // If so, use the following code to get the interpolated z value.
                auto [alpha, beta, gamma] = computeBarycentric2D(m + 0.75, n + 0.25, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(m, n);
                if (z_interpolated < depth_buf_2[index]) {
                    depth_buf_2[index] = z_interpolated;
                    frame_buf_2[index] = t.getColor();
                }
            }
            if (insideTriangle(m + 0.75, n + 0.75, t.v))
            {
                // If so, use the following code to get the interpolated z value.
                auto [alpha, beta, gamma] = computeBarycentric2D(m + 0.75, n + 0.75, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(m, n);
                if (z_interpolated < depth_buf_3[index]) {
                    depth_buf_3[index] = z_interpolated;
                    frame_buf_3[index] = t.getColor();
                }
            }

        }
    }

    
}

```
效果对比：
![对比](/image/games101/assignment23.png)




## Lecture 07~09 Shading 着色

应用材质到物体上的过程称为着色。
着色的局限性：只考虑物体本身和光源，不考虑遮挡关系以及阴影生成
### Blinn-Phong光照模型
将光照分为环境光（Ambient）、漫反射光（Diffuse）和镜面高光（Specular），对复杂光照的近似和简化

### 漫反射 Diffuse Reflection
![漫反射](/image/games101/07-091.png)
* 光照方向l，法线n，观察方向v
* 兰伯特Lambert光照模型：一种漫反射光照模型
* $\mathbf{I}\mathit{  {\tiny diffuse} }  = \mathbf{I}\mathit{  {\tiny light} } \cdot \mathbf{K}\mathit{  {\tiny diffuse} } \cdot max（ 0 ， n\cdot l ）\cdot attenuation $
* 核心理论为漫反射强度与入射光与法线夹角相关（即为入射光越接近垂直入射，漫反射越强），与观察角度无关
* 分别为漫反射光照，光源光照，漫反射系数（材质决定），光照和法线夹角余弦（不取负值），光照衰减
* 课程中使用 $ \frac{1}{d^{2} } $的衰减系数，但是实际使用时往往使用 $ \frac{1}{a0+a1\cdot d +a2\cdot  d^{2} }  $作为衰减系数，便于进行细微调整

### 高光 Specular Highlight
* 当观察方向与光照反射方向接近时，认为能够接收到高光反射
* Phong式光照模型中利用光照方向 l 和法线 n 计算出反射光方向，并和观察方向进行比较
* Blinn-Phong光照模型使用光照方向 l 和观察方向 v 的半程向量 h 和法线 n 进行比较，优化计算
* 其中半角向量满足$ \mathbf{h} = \frac{l+v}{\left \| l+v \right \| } $
* $\mathbf{I}\mathit{  {\tiny specular} }  = \mathbf{I}\mathit{  {\tiny light} } \cdot \mathbf{K}\mathit{  {\tiny specular} } \cdot  max \left ( 0 , n\cdot h \right )^{shininess} \cdot attenuation$
* 分别为高光，光源光照，高光系数（材质决定），法线与半角向量夹角余弦（不取负值），光泽度（调整高光范围，因为一次方的cos的变化过慢，使得高光面积很大），光照衰减

### 环境光 Ambient Lighting
* Blinn-Phong光照模型中的环境光是恒定不变的常数，与光源方向、法线、观察方向都无关。主要是为了保证始终有一定的光照会照亮物体
$\mathbf{I}\mathit{  {\tiny ambientOut} }  = \mathbf{I}\mathit{  {\tiny ambientIn} } \cdot \mathbf{K}\mathit{  {\tiny ambient} }$
* 分别为环境光，环境入射光（可以使用主要光源的固定衰减值），环境光系数
![光照](/image/games101/07-092.png)


### 着色频率（Shading Frequncies）
![光照](/image/games101/07-093.png)
从左到右增加着色频率，从上到下增加模型面数

**逐三角面着色（也称Flat Shading）**

每一个面根据面法线执行一次光照计算，结果应用到整个三角面
计算量极低，每个面只进行一次光照计算
**逐顶点着色（也称Gouraud Shading）**

每个顶点根据顶点法线执行一次光照计算，面内部光照通过插值计算获得
计算量较低，每个面平均进行一次光照计算（并不严谨，实际受顶点数和面数影响），同时有插值运算开销
**逐像素/片元着色（也称Phong Shading）**

每个像素根据自身法线（法线根据顶点法线插值获得）独立进行光照计算并应用于自身
计算量较高，像素法线根据顶点法线插值获取，每个像素都进行一次光照计算
计算量大小并不绝对，当面数无限增加（比如数亿个面的高模模型）超过屏幕像素数量时，逐三角面着色反而效果最好计算量最大。

### 渲染管线（Rendering Pipeline）
![光照](/image/games101/07-094.png)

渲染管线，不同的资料中可能细节不同，但整体顺序类似
顶点处理（Vertex Processing）；处理顶点位置、顶点法线的MVP变换，可能存在的着色计算，一些特殊的变量计算（比如顶点的世界坐标，TBN矩阵等）
三角形处理（Triangle Processing）：将顶点按规则组装为三角面，通常曲面细分/顶点增减也在该阶段进行
光栅化（Rasterization）：根据三角形数据将三角面离散为片元/像素
片元/像素处理（Fragment Processing）：GPU会自动对顶点的数据进行插值运算（光栅化的功劳？），片元处理的目标是通过各种计算（光照、纹理、阴影、AO等）得到像素的最终着色值
帧缓冲操作（Framebuffer Operations）：得到最后的渲染画面
可编程渲染管线（Programmable Rendering Pipeline​）

现在通常指顶点着色器（Vertex Shader）和片元着色器（Fragment Shader）
Shader在GPU中执行，利用了GPU的高度并行性，每个像素都会执行shader，所以无需在Shader中编写循环
shadertoy: [shadertoy](https://www.shadertoy.com/view/ld3Gz2)

### 纹理映射（Texture Mapping）
* 将3D模型的表面映射到2D纹理图像中
* 纹理也就是常说的贴图，通常过程为顶点信息中带有纹理坐标（uv坐标）信息，片元的纹理坐标通过三角形重心插值获取，然后依据纹理坐标在纹理中进行采样获得最终的纹理颜色
* 纹理具有多种类型，最常见的就是漫反射纹理，其他的还有法线纹理、AO纹理、高光纹理、立方体纹理等

**重心坐标（Barycentric Coordinates）**

* 三角形插值时的计算方式
* 三角形内部的点都可以使用三个顶点的线性组合进行表示，且满足如下性质
$ P =\alpha A+\beta B+\gamma C $
$ \alpha +\beta +\gamma =1  $
$ \alpha \ge 0,  \beta \ge 0, \gamma \ge 0  $
* 重心坐标的问题：重心坐标不具有投影不变性，所以需要对3D空间中原始的位置进行插值运算后再投影，而不是对投影后的位置进行插值运算

**纹理映射的问题**
* 当纹理过小时（纹理本身也是一副图像），多个屏幕像素映射到纹理中同一像素中，导致产生纹理锯齿
* 当纹理过大时，较少的屏幕像素映射到较多的纹理像素中，使得纹理像素信息丢失，可能产生摩尔纹、画面断裂等问题


**解决方案**
纹理过小时：
最邻近算法（Nearest）：不进行处理，未落到纹理像素中心的坐标，直接使用所处像素的中心的颜色值
双线性插值（Bilinear）：对于未落到纹理像素中心的坐标，根据位置取周围4个纹理像素进行水平和竖直的线性插值获得最后的颜色值
双立方插值（Bicubic）：使用更多的纹理像素（比如16个）进行插值运算，以获得更好的效果
![双线性插值](/image/games101/07-095.png)
**双线性插值**

纹理过大时：
使用Mipmap技术，生成一系列的低分辨率图像（增加1/3的内存空间）
启用各向异性过滤（Anisotropic Filtering）
![Mipmap](/image/games101/07-096.png)
**Mipmap**
基于原图像生成一系列的低分辨率图像（增加1/3的内存空间）
Mipmap只能进行方形的近似范围查询
查询Mipmap层级的计算：
计算出当前像素在纹理中占据大小的近似值：取周围临近的像素，计算和当前像素的uv坐标差值，取最大的值认为是当前像素在纹理中占据大小的近似正方形边长 L
计算出应该查询的Mipmap层级：由于Mipmap以2倍的速度缩小，计算 D = \log_{2}{L} 即可
三线性插值：如果计算出的层数不为整数，对上下两个层级进行两次查询，然后在层间进行一次线性插值即可
![三线性插值](/image/games101/07-097.png)
L取最长的邻近像素uv距离，D = log2 L
**各向异性过滤（Anisotropic Filtering）**
Mipmap只能进行方形的范围查询，对于狭长的范围查询效果不好，这时可以通过各向异性过滤解决
各向异性过滤在Mipmap基础上生成额外生成了一系列横向压缩和纵向压缩的图像（总计增加3倍内存空间）
使得在范围查询上的效果更好
![各向异性过滤](/image/games101/07-098.png)
各向异性过滤生成的Mipmap图

### 纹理应用（Application of Texture）
**环境映射（Environment Map）**

* 使用一张纹理/贴图实现周围的环境光照（比如天空盒就常用立方体贴图实现）
* 近似认为环境光是无限远的，改变物体位置不会改变环境映射的结果（不考虑遮挡的前提下）
* 常用的有球面贴图（Spherical Map）和立方体贴图（Cube Map）
**凹凸贴图/法线贴图（Bump Map / Normal Map）**

凹凸贴图和法线贴图实际存在一定的区别。课程中讲解的意思似乎是可以通过凹凸贴图的深度变化计算出法线。
接下来的笔记主要关于法线贴图，实际中法线贴图也比凹凸贴图更加常用
常用的法线贴图通常为蓝色，比如下图
![Brick_Normal](/image/games101/07-099.png)
Brick_Normal
这主要是因为这样的法线贴图是基于顶点的切线空间的，实际使用时需要根据顶点的切线（Tangent）和法线（Normal）计算出副切线（Bitangent），组成TBN矩阵（Tangent、Bitangent、Normal）。然后使用TBN矩阵与法线贴图采样结果相乘并归一化得到最后的法线
由于通常法线都是对于原法线的微调，所以Normal（即Blue通道）的值比较大，所以法线贴图通常呈现为蓝色

**位移/置换贴图（Displacement Map）**
* 实际改变模型的顶点位置，得到更加真实的高模效果（支持自遮挡阴影，投射阴影等）

**程序纹理（Procedural Map）**
常用于木纹纹理、岩石纹理、山脉、年轮等2D或3D纹理中
常常使用噪声函数，比如著名的Perlin柏林噪声
不会生成具体的纹理图，而是通过函数确定不同位置的渲染结果

**环境光遮蔽纹理（AO Map）**
环境光遮蔽（Ambient Occlusion）指的是由于周围物体的遮蔽导致的环境光减弱的现象（比如凹陷处更暗，凸出处更亮）
记录预定义的AO强度，实际使用时直接进行采样而不需要进行实时计算。保证在低性能条件下也有较好的AO效果

## Assignment3

1. 修改函数 rasterize_triangle(const Triangle& t) in rasterizer.cpp: 在此
处实现与作业 2 类似的插值算法，实现法向量、颜色、纹理颜色的插值。
``` bash
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
    // TODO: From your HW3, get the triangle rasterization code.
    
    std::array<Eigen::Vector4f, 3> v = t.toVector4();
    
    // iterate through the pixel and find if the current pixel is inside the triangle
    int  xmin = std::min(std::min(v[0].x(), v[1].x()), v[2].x());
    int  ymin = std::min(std::min(v[0].y(), v[1].y()), v[2].y());
    int  xmax = std::max(std::max(v[0].x(), v[1].x()), v[2].x());
    int  ymax = std::max(std::max(v[0].y(), v[1].y()), v[2].y());

    for (int m = xmin; m <= xmax; m++)
    {
        for (int n = ymin; n <= ymax; n++)
        {
            if (insideTriangle(m+0.5,n+0.5,t.v))
            {
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(m + 0.5, n + 0.5, t.v);
                // TODO: Inside your rasterization loop:

                float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                zp *= Z;
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(m, n);
                if (zp < depth_buf[index]) {
                    //set_pixel(p, t.getColor());
                    depth_buf[index] = zp;
                    // TODO: Interpolate the attributes:
                    // auto interpolated_color
                    // auto interpolated_normal
                    // auto interpolated_texcoords
                    // auto interpolated_shadingcoords
                    auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                    auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                    auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                    auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                    // Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                    // Use: payload.view_pos = interpolated_shadingcoords;
                    // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
                    // Use: auto pixel_color = fragment_shader(payload);
                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;
                    auto pixel_color = fragment_shader(payload);
                    Vector2i vertex;
                    vertex << m, n;
                    set_pixel(vertex, pixel_color);
  
                }
 
            }

        }
    }

 
}
```
![法向量实现结果](/image/games101/assignment31.png)


修改函数 phong_fragment_shader() in main.cpp: 实现 Blinn-Phong 模型计
算 Fragment Color.
``` bash
.....

for (auto& light : lights)
{
    // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
    // components are. Then, accumulate that result on the *result_color* object.


    Eigen::Vector3f LightDir = (light.position - point).normalized();
    float r2 = (light.position - point).squaredNorm();
    Eigen::Vector3f ViewDir = (eye_pos - point).normalized();
    Eigen::Vector3f halfDir = (ViewDir+LightDir).normalized();
    float NoL = normal.dot(LightDir);
    float NoH = normal.dot(halfDir);
    //diffusecolor 
    Eigen::Vector3f diffuseColor = kd *(light.intensity[0]/ r2)* std::max(0.0f, NoL);

    //specularcolor 
    Eigen::Vector3f specularColor = ks * (light.intensity[0] / r2) * std::pow(std::max(0.0f, NoH),p);

    //ambientcolor 
    Eigen::Vector3f ambientColor = ka * amb_light_intensity[0];


    result_color += diffuseColor+ specularColor+ ambientColor;
    
}

.....


```
![Blinn-Phong](/image/games101/assignment32.png)

修改函数 texture_fragment_shader() in main.cpp: 在实现 Blinn-Phong
的基础上，将纹理颜色视为公式中的 kd，实现 Texture Shading Fragment
Shader.
``` bash
.....
   if (payload.texture)
   {
       // TODO: Get the texture value at the texture coordinates of the current fragment

       return_color = payload.texture->getColor(payload.tex_coords[0], payload.tex_coords[1]);

   }
.....

```
![texture_fragment_shader](/image/games101/assignment33.png)

修改函数 bump_fragment_shader() in main.cpp: 在实现 Blinn-Phong 的
基础上，仔细阅读该函数中的注释，实现 Bump mapping.

``` bash

.....
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)
    Eigen::Vector3f n;
    n << normal;
    float x = normal[0];
    float y = normal[1];
    float z = normal[2];
    Eigen::Vector3f t;
    t << x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z* y / sqrt(x * x + z * z);
    Eigen::Vector3f b;
    b << n.cross(t);
    Eigen::Matrix3f TBN;
    TBN <<
        t.x(), b.x(), n.x(),
        t.y(), b.y(), n.y(),
        t.z(), b.z(), n.z();
    float u = payload.tex_coords[0];
    float v = payload.tex_coords[1];
    float w = static_cast<float>(payload.texture->width);
    float h = static_cast<float>(payload.texture->height);
    float dU = kh * kn * (payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
    float dV = kh * kn * (payload.texture->getColor(u, v + 1 / h).norm() - payload.texture->getColor(u, v).norm());
    Eigen::Vector3f ln;
    ln << -dU, -dV, 1;
    point += kn * n * payload.texture->getColor(u, v).norm();
    normal= (TBN * ln).normalized();

.....



```
![bump_fragment_shader](/image/games101/assignment34.png)


修改函数 displacement_fragment_shader() in main.cpp: 在实现 Bump
mapping 的基础上，实现 displacement mapping.

``` bash
.....
    float kh = 0.2, kn = 0.1;

    Eigen::Vector3f n;
    n << normal;
    float x = normal[0];
    float y = normal[1];
    float z = normal[2];
    Eigen::Vector3f t;
    t << x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z* y / sqrt(x * x + z * z);
    Eigen::Vector3f b;
    b<< n.cross(t);
    Eigen::Matrix3f TBN;
    TBN << 
        t.x(), b.x(), n.x(),
        t.y(), b.y(), n.y(),
        t.z(), b.z(), n.z();
    float u = payload.tex_coords[0];
    float v = payload.tex_coords[1];
    float w = static_cast<float>(payload.texture->width);
    float h = static_cast<float>(payload.texture->height);
    float dU = kh * kn * ( payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
    float dV = kh * kn *(payload.texture->getColor(u, v + 1 / h).norm() - payload.texture->getColor(u, v).norm());
    Eigen::Vector3f ln;
    ln <<-dU, -dV, 1;

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = (TBN * ln).normalized();
.....


```
![displacement_fragment_shader](/image/games101/assignment35.png)


双线性纹理插值: 使用双线性插值进行纹理采样, 在 Texture
类中实现一个新方法 Vector3f getColorBilinear(float u, float v) 并
通过 fragment shader 调用它。为了使双线性插值的效果更加明显，你应该
考虑选择更小的纹理图。请同时提交纹理插值与双线性纹理插值的结果，并
进行比较。

``` bash
    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        // 1. Clamp UV坐标到[0,1]范围
        u = std::max(0.0f, std::min(1.0f, u));
        v = std::max(0.0f, std::min(1.0f, v));
        
        auto u_img = u * width;
        auto v_img = (1 - v) * height;

        auto u0 = std::floor(u_img);
        auto u1 = u0 +1;
        auto v0 = std::floor(v_img);
        auto v1 = v0 +1;

        Eigen::Vector3f c01 = getColor(u0 /width, v1 / height);
        Eigen::Vector3f c11 = getColor(u1 / width, v1 / height);
        Eigen::Vector3f c00 = getColor(u0 / width, v0 / height);
        Eigen::Vector3f c10 = getColor(u1 / width, v0 / height);
        
        float s = (u_img - u0)/width;
        float t = (v_img - v0) / height;

        Eigen::Vector3f color0 = c00 + (c10- c00) * s;
        Eigen::Vector3f color1 = c01 + (c11-c01) * s;
        Eigen::Vector3f color = color0 * (1 - t) + color1 * t;
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

```
![Point and Bilinear](/image/games101/assignment36.png)

注意：uv要限制在0-1内
``` bash
    Eigen::Vector3f getColor(float u, float v)
{
    auto u_img = u * width;
    auto v_img = (1 - v) * height;
    if (u <= 0 || u >= 1 || v <= 0 || v >= 1)
        return Eigen::Vector3f(0, 0, 0);
    auto color = image_data.at<cv::Vec3b>(v_img, u_img);

    return Eigen::Vector3f(color[0], color[1], color[2]);
}
```
## Lecture 10~12 Geometry 几何
### 隐式几何（Implicit Geometry）
使用函数式的方式表示一个几何体
优点：很容易判断某个点是否在几何体上
缺点：通过函数式往往不能知晓几何体的具体形状
![隐式几何](/image/games101/10-121.png)
复杂的函数式很难看出对应的几何体形状

### 其他隐式几何
**CSG（Constructive Solid Geometry）**
复杂的几何体可以通过简单几何体的集合运算（Boolean Operation）得到
![CSG](/image/games101/10-122.png)
Combine implicit geometry via Boolean operations

**距离函数（Distance Fuction）**
描述点到几何体表面的最近距离（内部为负值）
与有向距离场SDF（Signed Distance FIeld）的概念强相关
![小球融合](/image/games101/10-123.png)
小球融合
![2D的距离场融合](/image/games101/10-124.png)
2D的距离场融合

**分形（Fractals）**
自相似的递归图形（比如雪花末端的子结构相似性）
渲染中容易引起严重的走样问题

### 显式几何（Explicit Geometry）
通过参数映射表示的几何体/直接给出的几何体（参数映射：一张纹理，其uv坐标都有对应的xyz坐标，所以可以通过参数映射重建出3D模型）
优点：可以清晰地得到几何体的形状信息
缺点：难以判断某个点和几何体的位置关系
**其他显式几何**
点云（Point Cloud）
一堆密集的点的集合，只有当点的密度足够时才能进行有效显示（比如3D扫描的数据通常是点云的形式）
![点云](/image/games101/10-125.png)
点云形成的古罗马斗兽场

多边形网格（Polygon Mesh）
应用最广泛的显式几何表示方法
将面拆解为多边形（通常为三角形或四边形）进行显示
常见的3D模型通常都是这种显示方式
![网格](/image/games101/10-126.png)
曲线和曲面（Curves and Surfaces）也是显式的几何表示方式

### 贝塞尔曲线（Bezier Curves）
**贝塞尔曲线的定义**

由若干控制点控制生成的曲线
曲线必须经过第一个和最后一个控制点
在起始位置和结束位置曲线的方向必须与最接近的边相同
![三点贝塞尔](/image/games101/10-127.png)
三个控制点的贝塞尔曲线
![四点贝塞尔](/image/games101/10-128.png)
四个控制点的贝塞尔曲线

实际上就是一个递归的线性插值过程，得到最后一个点时即为贝塞尔曲线上的点
**贝塞尔曲线的特点**

具有仿射不变性：直接对贝塞尔曲线进行仿射变换和对控制点进行仿射变换后重新绘制一条贝塞尔曲线等价
不具有投影不变性
凸包性质：贝塞尔曲线一定位于控制点的凸包内部（凸包：包围目标点的最小凸多边形）
**逐段贝塞尔曲线（Piecewise Bezier Curves）**

存在多个控制点时，将控制点按照一定规则（比如4个为一组）的方式进行分割，使得曲线更加多变
通常首尾的控制点会和其他曲线共享，以保证曲线的连续性
**样条线（B-splines）和NURBs**

**贝塞尔曲面（Bezier Surfaces）**

基于贝塞尔曲线，将多条（比如4条）贝塞尔曲线上的 t 时刻的点作为新的控制点生成新的贝塞尔曲线
这样一个时间 t 就对应为一条贝塞尔曲线而不再是一个点，然后按时间 t 扫描即可得到贝塞尔曲面

### 网格（Meshes）
**网格操作（Mesh Operations）**
细分（Subdivision）：增加模型面数，常见的有Loop细分、Catmull-Clark细分
简化（Simplification）：减少模型面数，常见的有边坍缩（Edge Collapsing）
规范化（Regularization）：调整三角形使其更加接近于正三角形

**Loop细分**

第一步：生成新三角形，方式为取三角形三条边的中点作为新顶点在三角形内部生成一个新三角形，将原本的一个三角形时机地分割为了四个三角形
第二步：调整顶点位置，Loop细分将顶点分为新顶点和旧顶点，分别进行调整
新顶点： $V = \frac{3}{8} (A+B)+ \frac{1}{8} (C+D)$
旧顶点： $V = (1-n\cdot u)\cdot origin +u\cdot neighborsum$
![新顶点更新方式](/image/games101/10-129.png)
新顶点更新方式
![旧顶点更新方式](/image/games101/10-1210.png)
旧顶点更新方式

**Catmull-Clark细分**
Loop细分只能对三角形进行细分，Catmull-Clark细分可以作用于更加一般的情况
重要概念：奇异点——度数不为4的顶点
第一步：生成新面，方式为取面的中点和所有边的中点进行连线
在第一次细分时会增加 n 个奇异点（n为非四边形面的个数）
在后续细分时，奇异点数量不再变化
第二步：调整顶点位置
面顶点（Face Point）： $f = \frac{1}{4} ( v_1 + v_2 + v_3 + v_4 )$
边顶点（Edge Point）： $e = \frac{1}{4} ( v_1 + v_2 + f_1 + f_2 )$
旧顶点（Vertex Point）： 
$v = \frac{1}{16} ( f_1 + f_2 + f_3 + f_4 + 2( m_1 + m_2 + m_3 + m_4 )+4p)$
![三种顶点及其位置计算的参考顶点](/image/games101/10-1211.png)
三种顶点及其位置计算的参考顶点

**边坍缩（Edge Collapsing）**
将一条边坍缩为一个顶点
常常用于模型LOD（Level of Details）等应用中
二次误差度量（Quadric Error Metrics）
坍缩后新点的位置到关联平面距离的最小平方和（也是决定新点位置的依据）
根据二次度量误差的值，从小到大组成坍缩次序
每坍缩一次，需要对被影响的边重新计算二次误差度量
使用堆的数据结构存储即可
![边坍缩](/image/games101/10-1212.png)
边坍缩

## Assignment4

实现 de Casteljau 算法来绘制由 4 个控制点表示的 Bézier 曲线 (当你正确实现该
算法时，你可以支持绘制由更多点来控制的 Bézier 曲线)。
1. 考虑一个 p0, p1, ... pn 为控制点序列的 Bézier 曲线。首先，将相邻的点连接
起来以形成线段。
2. 用 t : (1 − t) 的比例细分每个线段，并找到该分割点。
3. 得到的分割点作为新的控制点序列，新序列的长度会减少一。
4. 如果序列只包含一个点，则返回该点并终止。否则，使用新的控制点序列并
转到步骤 1。

``` bash
cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    if (control_points.size() == 1) return control_points[0];
    std::vector<cv::Point2f> next_control_points = {};
    for (int i=0; i < control_points.size()-1; i++)
    {
        auto& a = control_points[i];
        auto& b = control_points[i + 1];
        auto p = a + t * (b - a);
        next_control_points.push_back(p);
    }
    return recursive_bezier(next_control_points, t);

}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.

    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        cv::Point2f point = recursive_bezier(control_points, t);

        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }

}
```
![Point and Bilinear](/image/games101/assignment41.png)


实现对 Bézier 曲线的反走样。(对于一个曲线上的点，不只把它对应于一个像
素，你需要根据到像素中心的距离来考虑与它相邻的像素的颜色。)

``` bash
void GetColorBezier(cv::Point2f Point, cv::Mat& window)
{
    auto centerPoint = cv::Point2f(int(Point.x) + 0.5f, int(Point.y) + 0.5f);

    auto x = Point.x, y = Point.y, centerX = centerPoint.x, centerY = centerPoint.y;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            float distance = std::sqrt(std::pow(x - (centerX - i), 2) + std::pow(y - (centerY - j), 2));
            float normal_distance = distance * std::sqrt(2) / 3;
            float color = 255.0f * (1 - normal_distance);

            if (window.at<cv::Vec3b>(centerY - j, centerX - i)[1] < color)
                window.at<cv::Vec3b>(centerY - j, centerX - i)[1] = color;
        }
}


```

![Point and Bilinear](/image/games101/assignment42.png)


## Lecture 13~16 Ray Tracing 光线追踪
### 阴影映射（Shadow Mapping）
光栅化过程中，着色（Shading）只考虑了光源和材质，并未考虑遮挡以及阴影的计算，所以在光栅化流程中，阴影需要单独解决。阴影映射（Shadow Mapping）即为光栅化流程的阴影解决方案
核心思想：摄像机能“看到”但是光源不能“看到”的位置即为阴影位置

**阴影贴图（Shadow Map）**
最简单的传统阴影贴图
步骤：
从光源处看向场景，渲染一副深度图Shadow Map
从摄像机看向场景，对看到的位置相对于光源的深度和Shadow Map中对应深度进行比较，相等则为明亮部分，不相等则为阴影部分
存在问题：
只能实现硬阴影
由于Shadow Map的分辨率问题，会导致阴影出现锯齿、错误自遮挡等问题
深度以浮点数的形式存储，存在浮点数精度问题

**级联阴影贴图 CSM（Cascaded Shadow Maps）**
传统的Shadow Map以光源处为视角，阴影离光源越近精度越高，离光源越远精度越低。但是理想中的效果应该是离摄像机距离越近阴影精度越高。
CSM以摄像机角度出发，根据Z值划分视锥体为不同层级，分别对光源生成阴影贴图，进而实现基于摄像机位置距离的阴影精度控制

### 光线追踪基础（Ray Tracing）
**使用光线追踪的原因**
光栅化不能很好地实现一些效果（比如软阴影、Glossy材质、间接光照等）
光栅化整体的光照质量相对而言比较差

**光线追踪的一些基础假设**
光沿直线传播（不考虑光的波动性，所以无法实现双缝干涉等现象）
光线之间不会发生碰撞
光线是从光源不断传播直到到达视锥体的（光路可逆）

**光线投射（Ray Casting）：只计算直接光照的最简单的光线追踪**
从观测点通过成像平面的每一个像素投射出一条光线
找到第一个接触点
将该点与光源连接，判断光源可见性
根据结果进行着色并写至像素

### 递归光线追踪 Recursive (Whitted-Style) Ray Tracing
基于光线投射，当打到第一个接触点时，根据条件进行反射/折射
反射/折射光线打到下一个接触点，并递归地进行反射/折射
对每一个接触点，都与光源进行连线并判断可见性
将所有的着色共享进行加和作为最后的着色结果（叠加时需要考虑能量损失，否则会图像过曝）
![Recursive Ray Tracing图解](/image/games101/13-161.png)
Recursive Ray Tracing图解

### 光线求交（Ray-Surface Intersection）
**光线表示**：$ \mathbf{r} \left ( t \right ) =  o +t\mathbf{d} $ 

**交点求取**
隐式几何：直接联立方程组求解
显式几何：以三角形网格模型为例，则是求三角面和光线的交点
方法一：先与平面求交，然后判断交点和三角形的关系
方法二：Morller Trumbore算法：射线三角相交算法。速度更快，使用了重心坐标的知识
解方程：$\mathbf{o}+t\mathbf{d}=(1-b_1-b_2)\mathbf{P}_0+b_1\mathbf{P}_1+b_2\mathbf{P}_2$
如果结果满足$ t,b_1,b_2>0 $且 $b_1+b_2<1$,那么光线与三角面相交
问题在于如果对所有三角面进行遍历，那么开销将会非常大，所以需要进行加速

### 光线求交加速（Accelerating Ray-Surface Intersection）
包围盒（Bounding Box）：使用一个包围盒包围几何体，如果不与包围盒相交，那么一定不会与几何体相交

**包围球 Bounding Sphere**
最快的检测速度
紧密性最差（特别是对细长物体）

**方向包围盒 OBB（Oriented Bounding Box）**
最慢的检测速度
最好的紧密性

**轴对齐包围盒 AABB（Axis-Aligned Bounding Box）**
较快的检测速度
较差的紧密性
![轴对齐包围盒](/image/games101/截图_20250928124318.png)

**均匀网格（Uniform Grids）**
将空间分为均匀的网格
遍历物体，向占据的对应网格中添加信息
进行求交运算，将网格视为包围盒
无需对每一个网格进行判断，
存在问题：网格的大小需要精心设置，过大则起不到加速作用，过小则会空网格过多

**空间划分（Spatial Partition）**
四叉树/八叉树（Quadtree/Octree）：进行四分（2D）/八分（3D）平分生成子空间
KD-Tree：进行二分生成子空间，且分割轴以xyz轴的顺序进行
BSP-Tree：进行二分生成子空间，且分割轴任意
存在问题：
同一物体存在于多个包围盒中时，会重复进行计算
有的三角面难以判断是否在划分的包围盒中（比如3D空间中包围盒的一个角插入三角面中的情况）

**物体划分（Object Partition）**
包围体层次结构 BVH（Bounding Volume Hierarchy）
将包围盒按将物体分为两个部分计算出两个新的包围盒。递归执行直到达到结束标准（比如物体少于5个）
优点：按对象划分，物体一定在且只在一个包围盒中
常用技巧：
划分时，始终以当前包围盒最长的轴为标准进行划分
划分时通常以找寻中位数的方式确定划分标准


### 辐射度量学（Basic Radiometry）
**更精确地描述光线的方式**

光照属性

Energy 能量： $Q\mathrm{~[J=Joule]}$
Flux 通量：单位时间内的能量（即为功率）
$
\Phi=\frac{dQ}{dt}\quad\text{[W=Watt] [lm=lumen]}
$
Intensity 强度：光源发出的单位立体角内的通量
立体角（Solid Angle）：二维中单位弧度角$1/2\pi$的三维拓展，单位立体角为$1/4\pi$
$
I=\frac{\mathrm{d}\Phi}{\mathrm{d}\Omega}=\frac{\Phi}{4\pi}\quad\mathrm{[W/sr]~[lm/sr=cd=candela]}
$
Irradiance 辐照度：接收物上单位面积接收到的通量，
$
E=\frac{\mathrm{d}\Phi}{\mathrm{d}A}\quad\mathrm{[W/m^{2}][lm/m^{2}=lux]}
$
兰伯特余弦定律（Lambert's Cosine Law）：辐照度与入射方向和法线的夹角余弦成正比（即为Blinn-Phong光照模型中的漫反射光照公式的规则）
Radiance 辐亮度：光线在单位立体角单位投影面积上的通量（计算光照衰减）
$
L=\frac{\mathrm{d}^{2}\Phi}{\mathrm{d}\Omega\mathrm{~d}A\cos\theta}\quad[\mathrm{W}/(\mathrm{sr}\cdot\mathrm{m}^{2})][\mathrm{lm/(sr}\cdot\mathrm{m}^{2})=\mathrm{nit}]
$

为什么 $\boldsymbol{L}$中有 $cos\theta$修正而$\boldsymbol{E}$ 中没有： $\boldsymbol{E}$的公式为宏观统计结果，其中隐含了$cos\theta$ 的计算，如果使用辐亮度积分的方式进行计算，则需要显式包含$cos\theta$的计算

**双向反射分布函数 BRDF（Bidirectional Reflectance Distribution Function）**
分布函数（BRDF）：
描述表面某一点在入射方向 $\omega_{i}$的光线照射下，反射到出射方向 $\omega_{o}$的辐射亮度比例
$
f_r(\omega_i\to\omega_o)=\frac{\mathrm{d}L_o(\omega_o)}{\mathrm{d}E_i(\omega_i)}=\frac{\mathrm{d}L_o(\omega_o)}{L_i(\omega_i)\cos\theta_i\mathrm{~d}\omega_i}
$
互易性（Reciprocity）：$f_r(\omega_i\to\omega_o)=f_r(\omega_o\to\omega_i)$
反射方程（Reflection Equation）：
对于某一个固定的方向，将其所有入射方向光线的贡献相加/积分即可得到该出射方向的光线
$
L_o(\omega_o)=\int_{\Omega}f_r(\omega_i\to\omega_o)L_i(\omega_i)\cos\theta_i\mathrm{d}\omega_i
$
渲染方程（Rendering Equation）：
反射光的最终值即为自发光+反射光
$
L_o(\omega_o)=\underbrace{L_e(\omega_o)}_{\text{自发光}}+\int_\Omega\underbrace{f_r(\omega_i,\omega_o)}_{\mathrm{BRDF}}\underbrace{L_i(\omega_i)}_{\text{入射光}}\underbrace{\cos\theta_i}_{\text{投影修正}}\mathrm{~d}\omega_i
$
**不同的光照分类**

直接光照：光线经过一次弹射到达观测点
间接光照：光线经过一次以上弹射到达观察点
全局光照：0次到n次弹射的所有光线效果的总和

### 蒙特卡洛路径追踪（Monte Carlo Path Tracing）
**蒙特卡洛积分（Monte Carlo Integration）**
求  $\left[a,b\right]$ 区间内的定积分 $ I=\int_a^bf(x)\mathrm{~d}x $ 的一种方法
在 $\left[a,b\right] $ 积分域中随机地取x值，得到对应y值
将y值视为高度对积分域求面积作为定积分的结果
进行多次采样，求平均值作为最后结果
采样的依据为选定的概率密度函数 PDF（Probability Density Fuctions）
$
I\approx\underbrace{\frac{1}{N}}_\text{归一化}\sum_{i=1}^N\underbrace{f(x_i)}_{\text{样本贡献}},\quad\underbrace{x_i\sim p(x)}_{\text{采样分布}}
$
比如最简单的当PDF为均匀分布时，即为均匀随机采样
$
I\approx\frac{b-a}{N}\sum_{i=1}^Nf(x_i),\quad x_i\sim\mathcal{U}(a,b)
$
可以通过选取与积分函数 $f(x)$形状相似的 $p(x)$进行重要性采样
**路径追踪（Path Tracing）**
Whitted-Style光线追踪存在的问题：
生成次级光线时总是进行完美的镜面反射/折射，所以无法实现Glossy材质
光线打到漫反射材质后无法继续反射（因为无法确定光线的来源方向）
对渲染方程使用蒙特卡洛积分（忽略自发光项）
$
L_o\approx\frac{1}{N}\sum_{i=1}^N\frac{f_r(\omega_i,\omega_o)L_i(\omega_i)\cos\theta_i}{p(\omega_i)},\quad\omega_i\sim p(\omega)
$
均匀半球采样
最简单的采样方式
概率密度函数 $p(\omega)=\frac{1}{2\pi}$
为了避免指数爆炸，生成次级光线时只会采样一个方向。noise的问题由从像素打出多条光线path，将结果进行平均来减轻。(一个像素打出的光线数又称为SPP samples per pixel)
![光线数量指数爆炸](/image/games101/截图_20250928132105.png)
光源重要性采样
当光源比较小时，随机的采样很可能无法采样到光源，因此需要对光源进行独立的重要性采样
这时生成次级光线可能变为两条（一条显式采样光源，一条采样其他情况，因为采样光源的次级光线一定终止，所以不会导致指数爆炸）
需要判断光源可见性
递归问题修正：（俄罗斯轮盘赌）
路径追踪通过递归的方式求取间接光照，但是会出现无限递归的问题
可以通过限制递归深度解决无限递归，但是这样会使得能量出现损失
使用俄罗斯轮盘赌（Russian Roulette）解决该问题
规定一个概率 $P$ ，每次生成次级光线时有概率$P$生成光线，有概率 $1-P$不生成光线。
如果生成了光线，将得到的返回结果除以 $P$
这样，既解决了无限递归的问题，又在期望上满足了无限递归的能量，没有能量损失
$
\underbrace{P\cdot\left(\frac{L_o}{P}\right)}_{\text{概率 }P\text{ 的贡献}}+\underbrace{(1-P)\cdot0}_{\text{概率 }1-P\text{ 的贡献}}=L_o
$
拓展问题：闫老师最后提到的Games101课程不会提及但是同样很重要的问题

如何实现均匀采样，使得采样点在半球上均匀分布？
选择什么样的pdf进行采样（重要性采样）？
均匀半球采样（Uniform Hemisphere Sampling）
PDF： $p(\omega)=\frac{1}{2\pi}$
实现简单，适用于所有BRDF。缺点在于方差很大
光源重要性采样（Light Importance Sampling）
余弦加权采样（Cosine-weighted Sampling）
基于BRDF的采样（BRDF Importance Sampling）
随机数重要吗？低差异序列（Low Discrepancy Sequences）
MIS（Multiple Important Sampling）多重重要性采样
像素不同Path之间简单平均还是需要加权（Pixel Reconstruction Filter）
伽马矫正（Gamma Correction）
HDR图（High Dynamic Range）
与SDR（Standard Dynamic Range）相对应

## Assignment5
![alt text](/image/games101/截图_20250929163043.png)

在这部分的课程中，我们将专注于使用光线追踪来渲染图像。在光线追踪中
最重要的操作之一就是找到光线与物体的交点。一旦找到光线与物体的交点，就
可以执行着色并返回像素颜色。在这次作业中，我们需要实现两个部分：光线的
生成和光线与三角的相交。本次代码框架的工作流程为：
1. 从 main 函数开始。我们定义场景的参数，添加物体（球体或三角形）到场景
中，并设置其材质，然后将光源添加到场景中。
2. 调用 Render(scene) 函数。在遍历所有像素的循环里，生成对应的光线并将
返回的颜色保存在帧缓冲区（framebuffer）中。在渲染过程结束后，帧缓冲
区中的信息将被保存为图像。
3. 在生成像素对应的光线后，我们调用 CastRay 函数，该函数调用 trace 来
查询光线与场景中最近的对象的交点。
4. 然后，我们在此交点执行着色。我们设置了三种不同的着色情况，并且已经
为你提供了代码。
你需要修改的函数是：
• Renderer.cpp 中的 Render()：这里你需要为每个像素生成一条对应的光
线，然后调用函数 castRay() 来得到颜色，最后将颜色存储在帧缓冲区的相
应像素中。
• Triangle.hpp 中的 rayTriangleIntersect(): v0, v1, v2 是三角形的三个
顶点，orig 是光线的起点，dir 是光线单位化的方向向量。tnear, u, v 是你需
要使用我们课上推导的 Moller-Trumbore 算法来更新的参数。
``` bash
bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    // TODO: Implement this function that tests whether the triangle
    // that's specified bt v0, v1 and v2 intersects with the ray (whose
    // origin is *orig* and direction is *dir*)
    // Also don't forget to update tnear, u and v.
    Vector3f  E1 = v1- v0;
    Vector3f  E2 = v2 - v0;
    Vector3f   S = orig - v0;
    Vector3f   S1 = crossProduct(dir, E2);
    Vector3f   S2 = crossProduct(S, E1);
    float s1e1 = dotProduct(S1, E1);
    float s2e2 = dotProduct(S2, E2);
    float s1S = dotProduct(S1, S);
    float s2dir = dotProduct(S2, dir);

    float t =( 1/ s1e1 )* s2e2;
    float b1 = (1 / s1e1 )* s1S;
    float b2 = (1 / s1e1) * s2dir;
    if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0)
    {
        tnear = t;
        u = b1;
        v = b2;
        return true;
    }

    return false;
}

void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = std::tan(deg2rad(scene.fov * 0.5f));
    float imageAspectRatio = scene.width / (float)scene.height;

    // Use this variable as the eye position to start your rays.
    Vector3f eye_pos(0);
    int m = 0;
    for (int j = 0; j < scene.height; ++j)
    {
        for (int i = 0; i < scene.width; ++i)
        {
            // generate primary ray direction
            float x = 0.0f;
            float y = 0.0f;
            // TODO: Find the x and y positions of the current pixel to get the direction
            // vector that passes through it.
            // Also, don't forget to multiply both of them with the variable *scale*, and
            // x (horizontal) variable with the *imageAspectRatio*   
            x = (((i+0.5)/ scene.width)*2-1)* scale * imageAspectRatio;
            y = (1-((j + 0.5) / scene.height) * 2 ) * scale;
            //float z = scene.width * 1 / 2 ;

                   

            Vector3f dir = Vector3f(x, y, -1); // Don't forget to normalize this direction!
            framebuffer[m++] = castRay(eye_pos, dir, scene, 0);
        }
        UpdateProgress(j / (float)scene.height);
    }

    // save framebuffer to file
    std::string outputPath = Utils::PathFromAsset("output/assigment5.ppm");
    FILE* fp = fopen(outputPath.c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (char)(255 * clamp(0, 1, framebuffer[i].x));
        color[1] = (char)(255 * clamp(0, 1, framebuffer[i].y));
        color[2] = (char)(255 * clamp(0, 1, framebuffer[i].z));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}


``` 

## Assignment6
在之前的编程练习中，我们实现了基础的光线追踪算法，具体而言是光线传
输、光线与三角形求交。我们采用了这样的方法寻找光线与场景的交点：遍历场景
中的所有物体，判断光线是否与它相交。在场景中的物体数量不大时，该做法可以
取得良好的结果，但当物体数量增多、模型变得更加复杂，该做法将会变得非常低
效。因此，我们需要加速结构来加速求交过程。在本次练习中，我们重点关注物体
划分算法 Bounding Volume Hierarchy (BVH)。本练习要求你实现 Ray-Bounding
Volume 求交与 BVH 查找。
首先，你需要从上一次编程练习中引用以下函数：
• Render() in Renderer.cpp: 将你的光线生成过程粘贴到此处，并且按照新框
架更新相应调用的格式。
• Triangle::getIntersection in Triangle.hpp: 将你的光线-三角形相交函数
粘贴到此处，并且按照新框架更新相应相交信息的格式。
在本次编程练习中，你需要实现以下函数：
• IntersectP(const Ray& ray, const Vector3f& invDir,
const std::array<int, 3>& dirIsNeg) in the Bounds3.hpp: 这个函数的
作用是判断包围盒 BoundingBox 与光线是否相交，你需要按照课程介绍的算
法实现求交过程。
• getIntersection(BVHBuildNode* node, const Ray ray)in BVH.cpp: 建
立 BVH 之后，我们可以用它加速求交过程。该过程递归进行，你将在其中调
用你实现的 Bounds3::IntersectP.
``` bash
inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir,
                                const std::array<int, 3>& dirIsNeg) const
{

    double tenter = -std::numeric_limits<double>::infinity();
    double texit = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; i++) {
        //求三个轴的tmin,tmax
        // invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z),
        double tmin = (pMin[i] - ray.origin[i]) * invDir[i];
        double tmax = (pMax[i] - ray.origin[i]) * invDir[i];
        //用dirIsNeg判断光线方向
        if (!dirIsNeg[i])//如果i<0,则在i轴光线方向为负，则从pmax进入，pmin离开，swap tmin和tmaxx
            std::swap(tmin, tmax);
        tenter = std::max(tenter, tmin);
        texit = std::min(texit, tmax);
    }
    return tenter < texit && texit >= 0;



}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
    std::array<int, 3> dirIsNeg = { (ray.direction.x > 0), (ray.direction.y > 0), (ray.direction.z > 0) };
    Intersection isect;
    isect.happened = node->bounds.IntersectP( ray, ray.direction_inv, dirIsNeg);
    if (!isect.happened)
        return isect;
    if (node->object != nullptr)
    {
        return node->object->getIntersection(ray);
    }

    Intersection isect1, isect2;
    isect1 = getIntersection(node->left, ray);
    isect2 = getIntersection(node->right, ray);

    if (isect1.distance < isect2.distance)
        return isect1;
    else
        return isect2;


    //return Intersection{};
}




```
[加分项 20 points] SAH 查找：自学 SAH(Surface Area Heuristic) , 正
确实现 SAH 加速，并且提交结果图片，并在 README.md 中说明 SVH 的实现
方法，并对比 BVH、SVH 的时间开销。(可参考 http://15462.courses.cs
.cmu.edu/fall2015/lecture/acceleration/slide_024，也可以查找其他资
料)

``` bash

BVHBuildNode* BVHAccel::SAHrecursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    //假如物体数目过少 就直接使用原来的划分方法 没必要
    if (objects.size() <= 2) {
        node = recursiveBuild(objects);
        return node;
    }
    else {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        //计算索引
        constexpr int Bucketnum = 8;
        BucketInfo bucket[Bucketnum];
        for (int i = 0; i < objects.size(); i++)
        {
            int b = Bucketnum * centroidBounds.Offset(objects[i]->getBounds().Centroid())[dim];
            if (b == Bucketnum)
            {
                b = Bucketnum - 1;
            }
            bucket[b].count++ ;
            bucket[b].bounds = Union(bucket[b].bounds, objects[i]->getBounds());
        }
        //计算每个划分的代价
        float cost[Bucketnum-1];
        for (int i = 0; i < Bucketnum - 1; i++)
        {
            Bounds3 b0, b1;
            int count0 = 0,count1 = 0;
            for (int m =0;m<=i;m++)
            {
                b0 = Union(b0, bucket[m].bounds);
                count0 += bucket[m].count;
            }
            for (int n = i+1;n< Bucketnum;n++)
            {
                b1 = Union(b1, bucket[n].bounds);
                count1 += bucket[n].count;
            }
            cost[i] = 0.125f +
                (count0 * b0.SurfaceArea() +
                    count1 * b1.SurfaceArea()) /
                bounds.SurfaceArea();
        }
        //找出最小代价
        float minCost = cost[0];
        int minCostSplitBucket = 0;
        for (int i = 0; i < Bucketnum - 1; i++)
        {
            if (cost[i]<minCost)
            {
                minCost = cost[i];
                minCostSplitBucket = i;
            }
            
        }
        float leafCost = objects.size();
        int mid = 0;
        //如果包含的物体过多 或者缩小划分代价 则进行划分
        if (objects.size() > maxPrimsInNode|| minCost < leafCost)
        {
            for (int i = 0; i < objects.size(); ++i) {
                int b = Bucketnum *
                    centroidBounds.Offset(
                        objects[i]->getBounds().Centroid())[dim];
                if (b == Bucketnum) b = Bucketnum - 1;
                if (b <= minCostSplitBucket) {
                    std::swap(objects[i], objects[mid]);
                    mid++;
                }
            }
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + mid;
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = SAHrecursiveBuild(leftshapes);
        node->right = SAHrecursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}


```

## Assignment7
在之前的练习中，我们实现了 Whitted-Style Ray Tracing 算法，并且用 BVH
等加速结构对于求交过程进行了加速。在本次实验中，我们将在上一次实验的基
础上实现完整的 Path Tracing 算法。至此，我们已经来到了光线追踪版块的最后
一节内容。
请认真阅读本文档，按照本文档指示的流程完成本次实验。
``` bash

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    Vector3f L_dir;
    Vector3f L_indir;
    // TO DO Implement Path Tracing Algorithm here
    //从像素发出光线
    Intersection shadpoint = intersect(ray);
    if (!shadpoint.happened)
        return Vector3f(0,0,0);
    //打到光源
    if (shadpoint.m->hasEmission())
        return shadpoint.m->getEmission();

    //打到物体
    Vector3f p = shadpoint.coords;
    Material* m = shadpoint.m;
    Vector3f N = shadpoint.normal.normalized();
    Vector3f wo = ray.direction; // 像素到物体的向量

    // 有交点，对光源采样
    float pdf_light;
    Intersection lightpoint;
    sampleLight(lightpoint, pdf_light);

    Vector3f x = lightpoint.coords;
    Vector3f ws = (x - p).normalized(); //物体到光源
    Vector3f NN = lightpoint.normal.normalized();
    Vector3f emit = lightpoint.emit;
    float d = (x - p).norm();

    Ray ray_px(p, ws);
    Intersection blocked = intersect(ray_px);

    if (blocked.distance-d >-0.001)
    {

        Vector3f eval = m->eval(wo, ws, N);
        float cos_theta = dotProduct(N, ws);
        float cos_theta_x = dotProduct(NN, -ws);//ws从物体指向光源，与NN的夹角大于180
        L_dir  = emit * eval * cos_theta * cos_theta_x / std::pow(d, 2) / pdf_light;
    }
    float P_RR = get_random_float();
    if (P_RR < RussianRoulette) {
        Vector3f  wi = m->sample(wo, N);
        Ray ray_pwi(p, wi);
        Intersection inter = intersect(ray_pwi);
        if (inter.happened && !inter.m->hasEmission()) {
            Vector3f eval = m->eval(wo, wi, N);
            float pdf_O = m->pdf(wo, wi, N);
            float cos_theta = dotProduct(wi, N);
            L_indir = castRay(ray_pwi, depth + 1) * eval * cos_theta / pdf_O / RussianRoulette;

        }

    }

    return L_dir+ L_indir;
}

```
多线程：将多线程应用在 Ray Generation 上，注意
实现时可能涉及的冲突。
``` bash
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);
    int m = 0;

    // change the spp value to change sample ammount
    int spp = 16;
    std::cout << "SPP: " << spp << "\n";



    //for (uint32_t j = 0; j < scene.height; ++j) {
    //    for (uint32_t i = 0; i < scene.width; ++i) {
    //        // generate primary ray direction
    //        float x = (2 * (i + 0.5) / (float)scene.width - 1) *
    //                  imageAspectRatio * scale;
    //        float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

    //        Vector3f dir = normalize(Vector3f(-x, y, 1));
    //        for (int k = 0; k < spp; k++){
    //            framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / spp;  
    //        }
    //        m++;
    //    }
    //    UpdateProgress(j / (float)scene.height);
    //}


    std::mutex mutex;
    auto renderFunction = [&](uint32_t startRow, uint32_t endRow) {
        for (uint32_t j = startRow; j < endRow; ++j) {
            for (uint32_t i = 0; i < scene.width; ++i) {
                float x = (2 * (i + 0.5) / static_cast<float>(scene.width) - 1) *
                    imageAspectRatio * scale;
                float y = (1 - 2 * (j + 0.5) / static_cast<float>(scene.height)) * scale;

                Vector3f dir = normalize(Vector3f(-x, y, 1));
                Vector3f color(0, 0, 0);
                for (int k = 0; k < spp; ++k) {
                    color += scene.castRay(Ray(eye_pos, dir), 0) / spp;
                }

                std::lock_guard<std::mutex> lock(mutex);
                framebuffer[j * scene.width + i] = color;
            }
            std::lock_guard<std::mutex> lock(mutex);
            UpdateProgress(j / static_cast<float>(scene.height));
        }
        };

    unsigned numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    uint32_t rowsPerThread = scene.height / numThreads;

    for (unsigned t = 0; t < numThreads; ++t) {
        uint32_t startRow = t * rowsPerThread;
        uint32_t endRow = (t == numThreads - 1) ? scene.height : (t + 1) * rowsPerThread;
        threads.emplace_back(renderFunction, startRow, endRow);
    }

    for (auto& t : threads) {
        t.join();
    }



    UpdateProgress(1.f);

    // save framebuffer to file
    std::string outputPath = Utils::PathFromAsset("output/assigment7.ppm");
    FILE* fp = fopen(outputPath.c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
```


## Lecture 17 Materials and Appearances 材质和外观
材质和外观由BRDF决定，或者说材质就是BRDF
BSDF = BRDF + BTDF，分别为散射 Scattering，反射 Reflectance，透射 Transmittance
BRDF：纯反射模型，适用于不透明材质
BTDF：纯透射模型，适用于透明材质折射
BSDF：统一反射和透射，适用于复杂材质

### 基本材质
漫反射材质（Diffuse/Lambertian Material）
定义材质的反射率为$\rho=albedo$（体现对光的吸收/反射程度，范围为$\left[0,1\right]$）
当入射光均匀时（比如环境光），可以将渲染方程进行简化
$
L_o=\int_\Omega f_r\left.L_i(\omega_i)\right.\cos\theta_i\mathrm{~d}\omega_i=f_r\left.L_i\int_\Omega\right.\cos\theta_i\mathrm{~d}\omega_i=f_r\cdot L_i\cdot\pi
$
又因为$\rho=\frac{L_o}{L_i}$,可以得到Lamertian BRDF：$f_r=\frac{\rho}{\pi}$

Glossy材质（Glossy Material）
镜面反射/折射材质（Ideal Reflective/Refractive Material）
![Diffuse, Glossy, Reflect/Refact三种材质的图示](/image/games101/截图_20250928185248.png)

折射定律/斯涅尔定律（Snell's Law）
$
n_1\sin\theta_1=n_2\sin\theta_2\quad(\theta\text{为光线与法线夹角},n\text{为折射率})
$
全反射现象（Total Internal Reflection）
光线从一个折射系数大的介质射向一个折射系数小的介质时，只发生反射而不发生折射的现象
斯涅尔窗口（Snell's Window）：全反射现象的实际案例
![斯涅尔窗口，约97.2°](/image/games101/截图_20250928185500.png)

### 微表面材质（Microfacet Material）
菲涅尔项（Fresnel Term）

描述光线反射能量和透射能量占据比例的大小
绝缘体/电介质（Dielectric）：光线越垂直入射越倾向于透射，越水平入射越倾向于反射
导体（Conductor）：无论光线的入射角度如何，都比较倾向于反射
![alt text](/image/games101/截图_20250928185605.png)
菲涅尔项分为S偏振和P偏振，通常情况（非偏振光）取两者平均值
$
F_\mathrm{S}=\left(\frac{n_1\cos\theta_i-n_2\cos\theta_t}{n_1\cos\theta_i+n_2\cos\theta_t}\right)^2,\quad F_\mathrm{P}=\left(\frac{n_1\cos\theta_t-n_2\cos\theta_i}{n_1\cos\theta_t+n_2\cos\theta_i}\right)^2,F(\theta_i)=\frac{F_\mathrm{S}+F_\mathrm{P}}{2}
$
Schlick's approximation：对菲涅尔项的近似计算
$
F(\theta_i)\approx F_0+(1-F_0)(1-\cos\theta_i)^5\quad F_0=\left(\frac{n_1-n_2}{n_1+n_2}\right)^2
$
微表面材质（Microfacet Material）
将物体表面视为由大量的微小镜面组成，材质的区别由统计上微小镜面的法线分布决定
比如漫反射材质法线分布比较均匀，在各个方向都有；Glossy材质法线分布集中于宏观表面法线附近
![alt text](/image/games101/截图_20250928185829.png)
微表面材质BRDF（Microfacet BRDF）
$
f_r(i,o)=\frac{F(i,h)\left.G(i,o,h)\right.D(h)}{4\left(n\cdot i\right)\left(n\cdot o\right)}
$
分别为
菲涅尔项（Fresnel Term）：微表面模型由微小镜面组成，所以将半角向量 $h$视为菲涅尔项的法线
几何项/阴影遮挡项（Geometry/Shadow-Masking Term）：用于计算复杂微表面的自遮挡现象
法线分布项（Distribution of Normals）：描述与半角向量 $h$一致的微表面法线的宏观比例
### 各向同性/各向异性材质（Isotropic/Anisotropic Materials）
对于一个宏观表面，不改变入射方向和出射方向相对于宏观法线的俯仰角，只改变方位角，得到相同的BRDF的材质称为各向同性材质，否则称为各向异性材质。

各向同性材质：比如玻璃、光滑金属、普通漫反射材质等多数常见材质

各向异性材质：方向性打磨后的金属、织物、光盘表面等材质

### BRDF相关
**BRDF性质**
非负性（Non-negativity）： $f_{r}\geq0$
线性（Linearity）：BRDF可以进行线性的拆分和组合
可逆性（Reciprocity）： $f_r(\omega_r\to\omega_i)=f_r(\omega_i\to\omega_r)$
能量守恒：BRDF不会使得能量变得更多
各向同性/各向异性（Isotropic/Anisotropic）

**BRDF测量**
使用物理方法实际直接测量物体的BRDF
测量结果往往与公式推导相差较大
根据材质的各向同性/各向异性可以减少测量次数
BRDF材质库：MERL BRDF Database

## Lecture 18 Advanced Topics in Rendering 渲染前沿问题
### 高级光线传播（Advanced Light Transport）
无偏光线传播（Unbiased Light Transport）
* 蒙特卡洛的方法的估计值的期望与真实值相等
*  双向路径追踪 BDPT（）
*  Metropolis光线传播 MLT（）
有偏光线传播（Biased Light Transport）
*  估计值的期望和真实值不相等
*  若极限时可以收敛到真实值，则称为连续的（Cosistent）
*  光子映射（Photon Mapping）
*  VCM（Vertex Connection and Merging）

**双向路径追踪 BDPT（BiDirectional Path Tracing）**
从光源和摄像机分别打出光线作为“半路径”，连接两条半路径得到最终光路
在场景主要由间接光照照亮的情况下效果较好
![alt text](/image/games101/截图_20250928190620.png)

**Metropolis光线传播 MLT（Metropolis Light Transport）**
马尔科夫链蒙特卡洛 MCMC（Markov Chain Monte Carlo）：通过马尔科夫链生成概率密度函数pdf作为蒙特卡洛积分的采样pdf。使用该种方法生成的pdf形状和被积函数相似度比较高，效果较好。
通过对已有光线路径进行微小扰动，得到一条新的光线路径
在复杂场景中光线追踪效果较好
缺点：
难以估计渲染完成时间（无法确定收敛速度）
局部结果画面可能会显得“脏”（不同像素的收敛速度不同）
![alt text](/image/games101/截图_20250928190744.png)

**光子映射（Photon Mapping）**
原理：
第一步：从光源出发打出若干个光子，光子不断传播直到达到漫反射表面时停止
第二步：从摄像机出发打出光线，同样不断传播直到达到漫反射表面。统计此时位置周围最近的$n$个光子占据的面积（N邻近问题），根据面积大小决定该位置的亮度高低
在焦散（Caustics）渲染方面表现尤为出色
对于闫老师认为焦散翻译得不好的理解：类似于水底光纹的现象，能被人眼看到的亮度高的区域，从本质上来讲一定是光线聚焦到一起才会产生高亮度区域。

**VCM（Vertex Connection and Merging）**
结合双向路径追踪（BDPT）和光子映射（Photon Mapping）。
如果双向路径追踪中两条半路径不能连接，可以使用光子映射将一定范围内的光线进行融合，提高半路径利用率

**实时辐射度算法 IR（Instant Radiosity）**
也称为多光源方法（Many Light Approaches）
根据光源照亮的位置，将其视为新的虚拟点光源 VPL（Virtual Point Light）去照亮其他地方
![alt text](/image/games101/截图_20250928191318.png)

### 高级外观建模（Advanced Appearance Modeling）
**非表面模型（Non-surface models）**
散射/参与介质（Participating Media）
比如云/雾等，和体积渲染相关
通过相位函数（Phase Function）定义其向四周散射的方向和强度
![alt text](/image/games101/截图_20250928191455.png)

头发/毛发/纤维（Hair/Fur/Fiber）
根据毛发的真实生物结构，建立多层的材质模型
Kajiya-Kay Model：只计算反射光 $R$
Marschner Model：将头发视为玻璃柱，计算反射光 $R$、透射-透射光 $TT$、透射-反射-透射光 $TRT$
Double Cylinder Model：双层的柱状模型，增加了 $TTs$和$TRTs$ 两种通过髓质的光线模式
![Kajiya-Kay Model](/image/games101/截图_20250928191717.png)
Kajiya-Kay Model
![Marschner Model](/image/games101/截图_20250928191803.png)
Marschner Model
![Double Cylinder Model](/image/games101/截图_20250928191833.png)
Double Cylinder Model

颗粒材质（Granular Cylinder）
适用于比如沙子等粉状物体的材质
奥斯卡最佳动画短片

**表面模型（Suface models）**

透射/半透明材质（Translucent Material）
与有色玻璃的“半透明”概念不同，这里指的是玉石、皮肤、蜡烛等能一定程度透光的材质
BSSRDF 双向次表面散射分布函数（Subsurface Scattering）：对于BRDF的拓展，允许光线进入物体后经过一系列散射后从另一个点射出
可以使用两个光源近似次表面散射
![Dipole Approximation [Jensen et al. 2001]](/image/games101/截图_20250928192210.png)
Dipole Approximation [Jensen et al. 2001]
![alt text](/image/games101/截图_20250928192253.png)
布料（Cloth）
将布料视为物体表面，根据不同的织法使用不同的BRDF，效果一般但是简单快速
将布料视为散射介质，根据纤维的特性和分布构建散射分布函数
将布料的每一根纤维进行独立渲染，计算量大但是效果好
复杂材质（Detailed Material），Non-Statistical BRDF

**程序化生成（Procedural Appearance）**
通过噪声函数（比如Perlin noise），使得三维中的纹理不用创建，可以通过查询获得

## Lecture 19 Cameras, Lenses and Light Field 相机、透镜与光场
### 相机（Cameras）
**针孔相机（Pinhole Camera）**

针孔相机拍出的照片一定是锐利的（既没有虚化），也就是小孔成像不具有景深
视场 FOV（Field Of View）
$
FOV=2\mathrm{~}arctan(\frac{h}{2f})
$
$h$为传感器（sensor）的高度/宽度， $f$为焦距（Focal length）
为了方便统一，默认在计算时认为传感器的高度为35mm。（这解释了为什么手机可以拥有超出机身厚度的焦距，实际上是根据视场FOV的大小和默认的35mm高度的sensor计算得到。这里的焦距实际上是指该透镜对于35mm的传感器在该大小视场FOV下的等效焦距）
**曝光（Exposure）**
$
H=T\times E
$
其中 $H$为曝光量（Exposure）， $T$为曝光时间（Exposure Time）， $E$为辐照度（Irradiance）
**光圈（Aperture）**
控制单位时间内的进光量
光圈越大景深效果越明显
通过$F$描述光圈大小，具体表示为 $F_{n}$或者$F/n$。$n=\frac{f}{D}$ 即为焦距/光圈直径
**快门（Shutter）**
控制进光的时间长短，即为控制曝光时间
速度慢会使图像模糊
机械快门在拍摄高速物体时会出现扭曲现象
**感光度（ISO Gain）**
一种后期处理，用于调节画面明暗度
值较大时容易产生噪点

### 透镜（Lenses）
现实中的透镜由一组透镜组成，可以通过组合得到不同的焦距
Games101课程中使用理想薄透镜（即不计厚度且可以任意改变焦距）
透镜基础知识

平行光经过透镜后会聚焦于焦点，过焦点光经过透镜后会平行出射
$\frac{1}{f}=\frac{1}{z_o}+\frac{1}{z_i}$
$f$为焦距， $z_{o}$为相距， $z_{i}$为物距
![alt text](/image/games101/截图_20250928193214.png)

**弥散圆 CoC（Circle of Confusion）**

也称为模糊圈、散光圆
一个点的光线经过透镜后，聚焦点的位置位于成像平面之前或之后，导致一个点映射到一个圆面CoC
与人眼近似的原理类似
光圈越大，CoC越大。因为大光圈使得一个点射入透镜的光的范围变大，对于刚好聚焦的点无影响，但是对于映射到一个圆面的点，其CoC会变大。（这也是光圈越大，景深越明显的原因）

**景深（Depth of Field）**
场景中的光经过透镜打到成像平面。在成像平面附件的某个区域内，CoC都足够小，可以认为是清晰的，这个区域就被称为景深 DOF
![alt text](/image/games101/截图_20250928193330.png)

### 光场（Light Field）
**全光函数（The Plenoptic Function）**
$P(\theta,\varphi)$,通过方位角 $\theta$和俯仰角/极角 $\varphi$描述能够看到的所有方位的东西
$P(\theta,\varphi,\lambda)$,引入波长 $\lambda$，使得可以看到颜色
$P(\theta,\varphi,\lambda,t)$,引入时间 $t$，使得可以看到不同时间的场景
$P(\theta,\varphi,\lambda,t,x,y,z)$,引入位置$(x,y,z)$ ，使得可以在任意位置观察场景

**物体的光场**
光场是全光函数的一部分。
通过一个包围盒包围物体，然后对包围盒上每一个位置记录内部对该位置的光，这样就可以得到一个物体的光场包围盒
球面参数化（5D模型）
根据点的位置和光的方向定义光场，较为适用于物体的光场（支持全向的角度）
$P(\theta,\varphi,\lambda,t,x,y,z)$
![alt text](/image/games101/截图_20250928194117.png)

双平面参数化（4D模型）
通过两个平面上的点，相连得到一条光线方向定义光场，较为适用于光场摄像机（角度范围有限）
$P(\theta,\varphi,\lambda,t)$
![alt text](/image/games101/截图_20250928194506.png)

**光场摄像机（Light Field Camera）**
原理：通常摄像机记录光线的Irradiance，光场摄像机则是记录光线的radiance，允许后期对radiance进行自由组合得到不同的效果
优点：记录了完整的光场，可以后期进行自由调整
缺点：存储量更大，成本更高，分辨率通常不足

## Lecture 20 Color and Perception 颜色与感知
### 颜色基础
**物理学基础（Physical Basis）**
图形学中通常只关注光谱范围内的可见光（波长380~750nm）
谱功率密度 SPD（Spectral Power Distribution）
描述一束光在波长上的分布（一束光内部不同波长光的强度大小）
具有线性可加性（比如白光可以由单色光线性叠加得到）
![不同光线的SPD](/image/games101/截图_20250928195138.png)
不同光线的SPD

**生物学基础（Biological Basis）**

人眼中有两种感光细胞：视杆细胞和视锥细胞
视杆细胞（Rod Cells）：感知光的强度，可以得到灰度图
视锥细胞（Cone Cells）：感知光的颜色。有三种视锥细胞，可分为S、M、L细胞，对不同波长光线灵敏度不同
![三种视锥细胞对不同波长光的响应](/image/games101/截图_20250928195254.png)
颜色感知的结果为视锥细胞的感知曲线和光线的谱功率密度曲线的乘积
同色异谱（Metamerism）：不同SPD的光线可能经过和感知曲线的乘积后，得到相同的结果

### 色彩空间（Color Space）
加色系统和减色系统
加色系统：颜色叠加可以得到白色。
比如计算机中的RGB值，各种颜色光在白墙上的叠加
减色系统：颜色叠加可以得到黑色。
比如CMYK减色系统，颜料色彩叠加（原因是本质上是对吸收光的叠加）
sRGB（standardized RGB）
根据同色异谱现象，可以使用3种单色光RGB的不同强度组合匹配不同的颜色
sRGB的色域（gamut）受到一定的限制，无法混合出所有颜色
![Careful: these are not response curves or spectra!](/image/games101/截图_20250928195410.png)
Careful: these are not response curves or spectra!

CIE XYZ
一组假想的标准原色XYZ
Y是流明（Luminance），表示亮度，与颜色无关
由原色XYZ可以得到色度xyz

$
x=\frac{X}{X+Y+Z},y=\frac{Y}{X+Y+Z},z=\frac{Z}{X+Y+Z}
$
![CIE XYZ color matching functions](/image/games101/截图_20250928195518.png)
CIE XYZ color matching functions
![色域 Gamut](/image/games101/截图_20250928195556.png)
色域 Gamut

HSV/HSB（Hue-Saturation-Value/Brightness）
通过色调H、饱和度S、明度V/B定义色彩空间
CIE LAB
通过三个量，分别控制亮度和两对互补色（红绿、蓝黄）
CMYK
减色系统，由靛蓝、品红、黄色三种单色和黑色组成，广泛用于印刷

## Lecture 21~22 Animation and Simulation 动画与模拟
### 动画（Animation）
**关键帧动画（Keyframe animation）**

只定义关键帧信息，中间帧通过插值计算得到，然后连续播放帧即可实现关键帧动画
线性插值的效果往往不够好，需要应用一些平滑插值技术

**运动学（Kinematics）**

前向运动学（Forward Kinematics）
* 父关节带动子关节进行移动
* 根据父关节的旋转角度和子关节的旋转角度，从上到下进行旋转计算，得到最终结果
* 计算简单，但是操作不直观（需要定义旋转角度，难以直观地完成“移动到”某个位置的操作）
逆向运动学 IK（Inverse Kinematics）
* 直接控制末端关节的位置，其他位置通过计算得出
* 易于直观操作，但是计算更加复杂
* 可能出现解不唯一或者无解的问题。通过优化方法、梯度下降等方法解决问题，不直接依靠于数学求解

**绑定（Rigging）**
* 在模型上增加控制点，让人们可以方便的调整模型的动作
**动作捕捉（Motion Capture）**

* 在真人身上放置控制点捕捉运动数据
* 优点：快速捕捉大量数据，较为真实
* 缺点：复杂，昂贵；动作常常需要后期精修
* 常用方式：光学捕捉，磁力捕捉，机械捕捉

### 模拟（Simulation） 
**物理模拟（Physical Simulation）**

通过建立真实的物理模型，来模拟真实的动画效果
比如布料模拟和流体模拟
**质点弹簧系统（Mass Spring System）**
普通弹簧系统
* $f_{a\to b}=k_s\left.\frac{b-a}{||b-a||}\left(||b-a||\right.-l\right)$
* $l$为初始弹簧长度， $a,b$为弹簧两端， $k_{s}$为劲度系数
* 问题：因为没有阻尼力，会出现永远振荡的现象
阻尼力（Damping）
* $f_d=-k_d\mathrm{~}v$
* k_{d}为阻尼系数， $v$为速度
* 问题：会减慢所有速度，既会减慢振荡速度，也会减慢整体速度（比如受重力下落）
内部阻尼力（Internal Damping）
* $f_b=-k_d\left.\frac{b-a}{||b-a||}\cdot(v_b-v_a)\cdot\frac{b-a}{||b-a||}\right.$
* 解决了阻尼力会减慢整体速度的问题，现在只减慢振荡速度
弹簧网格
* 用网格状的弹簧模型模拟物体（比如布料）
斜向弹簧
* 解决弹簧网格中切变和不与边平行的对折问题
跳跃弹簧
* 解决弹簧网格中和与边平行的对折问题
![弹簧网格、斜向弹簧、跳跃弹簧](/image/games101/截图_20250928201932.png)
弹簧网格、斜向弹簧、跳跃弹簧
**粒子系统（Particle System）**

通过模拟每一个粒子的方式来模拟物体
比如灰尘、雾、水等物体可以使用粒子进行模拟
鸟群等集群算法也可以使用粒子进行模拟
**单粒子模拟（Single Particle Simulation）**
前向/显式欧拉方法（Forward/Explicit Euler Method）
* $\begin{aligned}
x_{t+\Delta t} & =x_t+\Delta t\cdot v_t \\
v_{t+\Delta t} & =v_t+\Delta t\cdot a_t
\end{aligned}
$
* 结果往往不准确且不稳定
![alt text](/image/games101/截图_20250928202142.png)
![alt text](/image/games101/截图_20250928202209.png)
中点方法（Midpoint Method）
具体步骤：
* 对$o$点使用显式欧拉方法得到下一个点$euler$点
* 取两点的中点为$mid$ 点，得到*euler*点，
* 回到 $o$点，使用 $v_{mid}$作为当前速度计算出下一个点$next$作为目标点
* $
\begin{aligned}
 & x_{t+\Delta t}=x_t+\frac{\Delta t}{2}\cdot(v_t+v_{t+\Delta t}) \\
 & v_{t+\Delta t}=v_t+\Delta t\cdot a_t \\
 & x_{t+\Delta t}=x_t+\Delta t\cdot v_t+\frac{\left(\Delta t\right)^2}{2}\cdot a_t
\end{aligned}
$
自适应步长（Adaptive Step Size）
* 对步长进行递归细分计算，直到某一次细分前后终点位置相差距离在可接受范围内
* 该方法会使得在不同的地方，自适应的选择合适的步长
反向/隐式欧拉方法（Backward/Implicit Euler Method）
* $
\begin{aligned}
 & x_{t+\Delta t}=x_t+\Delta t\cdot v_{t+\Delta t} \\
 & v_{t+\Delta t}=v_t+\Delta t\cdot a_{t+\Delta t}
\end{aligned}
$
龙格库塔方法（Runge-Kutta Families）
* 数值分析的相关方法，了解即可（反正我不大看得明白）
![alt text](/image/games101/截图_20250928202742.png)
* 基于位置/维尔莱积分（Position-based/Verlet Integration）
刚体模拟（Rigid Body Simulation）

* 不会发生形变，可以使用类似与单粒子模拟的方法
* 会增加一些参数，比如角度、角速度、角加速度等
流体模拟（Fluid Simulation）

* 由多个小的刚体球组成，且不可压缩或者拉伸
* 实现思路
    * 获取小球的分布位置，进而获取到小球密度
    * 根据密度和"正确密度"的差距，通过改变小球位置来使得小球密度趋近于"正确密度"
    * 加入能量损失，让流体可以停止运动
* 质点法/拉格朗日法 Lagrangian：逐个模拟每个粒子
* 网格法/欧拉法 Rulerian：将一个空间定义为一个网格，通过观察以网格为单位的时间如何变换来模拟
* 物质点方法 Material Point Method (MPM)：一种混合的方法 Hybrid


## Assignment8

``` bash
    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.

//        Comment-in this part when you implement the constructor
//        for (auto &i : pinned_nodes) {
//            masses[i]->pinned = true;
//        }
        Vector2D lengthvector = (end - start) / (num_nodes - 1);
        for (int i =0; i < num_nodes;i++)
        {
            Vector2D position = start + lengthvector * i;
            Mass* mass = new Mass(position, node_mass, false);
            masses.push_back(mass);
        }
        for (auto& i : pinned_nodes) {
                   masses[i]->pinned = true;
        }
        for (int i = 0; i < num_nodes-1; i++)
        {
            Spring* spring = new Spring(masses[i], masses[i+1], k);
            springs.push_back(spring);
        }

    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            Vector2D b_a= s->m1->position - s->m2->position;
            double b_alength = b_a.norm();
            Vector2D fb_a = s->k * (b_a / b_alength) * (b_alength - s->rest_length);
            s->m1->forces -= fb_a;
            s->m2->forces += fb_a;

            // Air Damping
            s->m1->forces += -0.005f * s->m1->velocity;
            s->m2->forces += -0.005f * s->m2->velocity;
        }
        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                m->forces += gravity * m->mass;
                Vector2D a = m->forces / m->mass;
                Vector2D velocityold = m->velocity;
                m->velocity = m->velocity + a * delta_t;
                m->position = m->position + m->velocity * delta_t;
                // TODO (Part 2): Add global damping

            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
            Vector2D b_a = s->m1->position - s->m2->position;
            double b_alength = b_a.norm();
            Vector2D fb_a = s->k * (b_a / b_alength) * (b_alength - s->rest_length);
            s->m1->forces -= fb_a;
            s->m2->forces += fb_a;
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                // TODO (Part 3.1): Set the new position of the rope mass
                // TODO (Part 4): Add global Verlet damping
                m->forces += gravity * m->mass;
                Vector2D a = m->forces / m->mass;
                float damping_factor = 0.00005;
                m->position = m->position+ (1- damping_factor)*(m->position - m->last_position) +a * delta_t * delta_t;
                m->last_position = temp_position;
                
            }
            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }
```