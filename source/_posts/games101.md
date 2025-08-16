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