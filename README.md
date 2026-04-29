# test4
1.  实验目标
- 理论理解： 理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）。
- 数学基础： 熟练掌握三维空间中的向量运算（法向量计算、光线方向、视线方向与反射向量）。
- 工程实践： 掌握如何利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数，直观感受各个参数对渲染结果的影响。
2.  实验原理
Phong 光照模型是一种经典的计算机图形学经验模型，它将物体表面反射的光分为三个独立的计算分量，最终将它们叠加得到像素颜色：
<img width="1647" height="70" alt="{2D48A3C4-0F20-47E5-B482-AF7670A1DC8A}" src="https://github.com/user-attachments/assets/8260d1e7-e0b9-4266-8158-c30e5587118e" />

- 环境光 (Ambient): 模拟场景中经过多次反射后均匀分布的背景光。
<img width="1612" height="83" alt="{ED781394-AE0F-4D89-B76B-24A5CDDDA336}" src="https://github.com/user-attachments/assets/ef573eb2-84ed-412d-a850-e18ecabba6ec" />

- 漫反射 (Diffuse): 模拟粗糙表面向各个方向均匀散射的光，强度与光线入射角的余弦值成正比（Lambert 定律）。

- 镜面高光 (Specular): 模拟光滑表面反射的强光，强度与观察方向和理想反射方向的夹角有关。

(注：$$\mathbf{N}$$为表面法向量，$$\mathbf{L}$$为指向光源的方向向量，$$\mathbf{V}$$为指向摄像机的方向向量，$$\mathbf{R}$$为光线的理想反射向量，$$n$$ 为高光指数 Shininess)
