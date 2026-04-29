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
<img width="1644" height="75" alt="{82D9AD5D-317E-4A41-8A06-C7C501286CBF}" src="https://github.com/user-attachments/assets/a72cdf63-610d-4dc4-ac80-7ddc3f0fe692" />

- 镜面高光 (Specular): 模拟光滑表面反射的强光，强度与观察方向和理想反射方向的夹角有关。
<img width="1702" height="196" alt="{E2A97D26-7241-4B8C-B546-A44515D9A35B}" src="https://github.com/user-attachments/assets/aa62efef-b7ea-4834-beda-e05cb52b576b" />

3.实验代码
import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# --- 几何体相交测试函数 ---

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """
    测试光线与竖直圆锥相交
    apex: 圆锥顶点坐标
    base_y: 圆锥底面的世界坐标 Y 值
    """
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到以顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建一元二次方程 At^2 + Bt + C = 0
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免 A 为 0 时的除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 保证 t_first 是较近的交点
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            # 验证交点是否在圆锥的高范围内 (局部 Y 坐标在 [-H, 0] 之间)
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 圆锥表面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 渲染红球 (放在左边)
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 渲染紫色圆锥 (放在右边)
        # 顶点在 y=1.2，底面在 y=-1.4
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 背景色
        color = ti.Vector([0.05, 0.15, 0.15]) 

        # 如果击中了任何物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # --- Phong 光照模型 ---
            ambient = Ka[None] * light_color * hit_color
            
            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color
            
            R = normalize(reflect(-L, N))
            spec = ti.max(0.0, R.dot(V)) ** shininess[None]
            specular = Ks[None] * spec * light_color 
            
            color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Phong Shading Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

    while window.running:
        # 执行并行渲染
        render()
        
        # 将渲染结果绘制到画布
        canvas.set_image(pixels)
        
        # 绘制交互面板
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()
4.实验结果

<img width="1280" height="850" alt="QQ20260429-093432" src="https://github.com/user-attachments/assets/0bc586f2-3c06-4f9d-9f21-5c7877dba6d0" />


5.选做部分代码
import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# --- 几何体相交测试函数 ---

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """
    测试光线与竖直圆锥相交
    apex: 圆锥顶点坐标
    base_y: 圆锥底面的世界坐标 Y 值
    """
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到以顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建一元二次方程 At^2 + Bt + C = 0
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免 A 为 0 时的除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 保证 t_first 是较近的交点
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            # 验证交点是否在圆锥的高范围内 (局部 Y 坐标在 [-H, 0] 之间)
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 圆锥表面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.func
def intersect_scene(ro, rd, exclude_t):
    """
    检测光线与场景中所有物体的相交，返回最近的交点距离
    exclude_t: 排除距离小于该值的交点（避免自相交）
    """
    t_min = 1e10
    hit = False
    
    # 1. 检测红球
    t_sph, _ = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    if 1e-5 < t_sph < t_min and t_sph != exclude_t:
        t_min = t_sph
        hit = True
    
    # 2. 检测圆锥
    t_cone, _ = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
    if 1e-5 < t_cone < t_min and t_cone != exclude_t:
        t_min = t_cone
        hit = True
        
    return hit, t_min

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 渲染红球 (放在左边)
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 渲染紫色圆锥 (放在右边)
        # 顶点在 y=1.2，底面在 y=-1.4
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 背景色
        color = ti.Vector([0.05, 0.15, 0.15]) 

        # 如果击中了任何物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # --- 阴影判断 ---
            # 阴影射线起点从交点稍微偏移一点，避免自相交
            shadow_ro = p + N * 1e-4
            shadow_rd = normalize(light_pos - shadow_ro)
            hit_shadow, _ = intersect_scene(shadow_ro, shadow_rd, min_t)
            
            # --- Blinn-Phong 光照模型 ---
            ambient = Ka[None] * light_color * hit_color
            
            # 阴影中只保留环境光
            if hit_shadow:
                color = ambient
            else:
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color

                # Blinn-Phong 高光计算
                H = normalize(L + V)  # 半程向量
                spec = ti.max(0.0, N.dot(H)) ** shininess[None]
                specular = Ks[None] * spec * light_color

                color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong & Hard Shadow Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

    while window.running:
        # 执行并行渲染
        render()
        
        # 将渲染结果绘制到画布
        canvas.set_image(pixels)
        
        # 绘制交互面板
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()

6.选做部分结果

<img width="1280" height="780" alt="QQ20260429-093538" src="https://github.com/user-attachments/assets/116b53bd-c3fe-4edb-a1fa-3b5b5f2a5c41" />

