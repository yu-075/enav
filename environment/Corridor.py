import random
import numpy as np
from typing import Iterable, Sequence, Tuple
import omni.kit.commands
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, UsdLux
from omni.isaac.core.utils.stage import get_current_stage

Color3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]

class MaterialHelper:
    def __init__(self, stage):
        """
        初始化材质助手
        Args:
            stage: USD 舞台对象 (Usd.Stage)
        """
        self.stage = stage

    def apply(self, prim, color: Sequence[float], name_suffix: str = "Material"):
        """
        为指定的 Prim 创建并绑定一个简单的 UsdPreviewSurface 材质
        Args:
            prim: 要绑定材质的 USD Prim
            color: RGB 颜色元组/列表 (范围 0-1)
            name_suffix: 材质名称后缀
        """
        path = prim.GetPath()
        material_path = Sdf.Path(f"{path}_{name_suffix}")
        material = UsdShade.Material.Define(self.stage, material_path)

        shader = UsdShade.Shader.Define(self.stage, material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        return material

WORLD_ENV_PATH = "/World/Env"
GROUND_PATH = "/World/GroundPlane"
LIGHT_PATH = f"{WORLD_ENV_PATH}/Lights/DistantLight"
WALL_COLOR = (0, 0.6, 0.6) # 墙体颜色（青色），与边界墙区分开
# 走廊参数（单位：米，按你的场景比例可再调整）
CORRIDOR_CLEAR_WIDTH = 0.4  # 两侧墙“内侧面”之间的净宽
WALL_THICKNESS = 0.1         # 墙体厚度（Cube size=1.0 下的 scale 值）
WALL_HEIGHT = 0.5
LEG_X_LENGTH = 20.0        # 第一段（沿 X）长度
TURN_X = 10.0              # 转弯点 X（与第一段右端对齐）
LEG_Y_LENGTH = 20.0        # 第二段（沿 +Y）长度
CAR_PATH = "/Leatherback"
CAR_USD_PATH = "/root/enav/car/leatherback.usd"
# 空间参数
SPACE_SIZE = 25.0  
#创建空间
def build_square_environment(stage) -> None:
    _remove_if_exists(stage, WORLD_ENV_PATH)
    _remove_if_exists(stage, GROUND_PATH)
    
    UsdGeom.Xform.Define(stage, WORLD_ENV_PATH)
    GroundPlane(prim_path=GROUND_PATH, z_position=0)
    mat_helper = MaterialHelper(stage)
    half_size = SPACE_SIZE / 2.0
    offset = half_size + (WALL_THICKNESS / 2.0)
    wall_len = SPACE_SIZE + WALL_THICKNESS
    z_height = WALL_HEIGHT / 2.0

    # 墙体颜色区分
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    # 1. 北墙 (+Y)
    create_wall(stage, mat_helper, "Wall_North", 
                pos=(0, offset, z_height), 
                scale=(wall_len, WALL_THICKNESS, WALL_HEIGHT), 
                color=colors[0])
    
    # 2. 南墙 (-Y)
    create_wall(stage, mat_helper, "Wall_South", 
                pos=(0, -offset, z_height), 
                scale=(wall_len, WALL_THICKNESS, WALL_HEIGHT), 
                color=colors[1])

    # 3. 东墙 (+X)
    create_wall(stage, mat_helper, "Wall_East", 
                pos=(offset, 0, z_height), 
                scale=(WALL_THICKNESS, wall_len, WALL_HEIGHT), 
                color=colors[2])

    # 4. 西墙 (-X)
    create_wall(stage, mat_helper, "Wall_West", 
                pos=(-offset, 0, z_height), 
                scale=(WALL_THICKNESS, wall_len, WALL_HEIGHT), 
                color=colors[3])

    # 灯光
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path(LIGHT_PATH))
    distant_light.CreateIntensityAttr(1000)

def _remove_if_exists(stage, prim_path: str) -> None:
    """
    如果指定路径的 Prim 存在，则将其从舞台中删除（用于重置环境）
    """
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def _as_vec3(value: Iterable[float]) -> Gf.Vec3f:
    """
    将输入转换为 Gf.Vec3f (单精度浮点向量)，通常用于 Scale 属性
    """
    x, y, z = value
    return Gf.Vec3f(float(x), float(y), float(z))


def _as_vec3d(value: Iterable[float]) -> Gf.Vec3d:
    """
    将输入转换为 Gf.Vec3d (双精度浮点向量)，通常用于 Translate 属性以匹配 API 要求
    """
    x, y, z = value
    return Gf.Vec3d(float(x), float(y), float(z))


def _world_min_max(stage, prim_path: str) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    计算并返回指定 Prim 在世界坐标系下的轴对齐包围盒 (AABB) 的最小值和最大值
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim 不存在或无效：{prim_path}")

    # 使用包围盒缓存读取世界空间 
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    world_bound = cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedBox()
    return aligned_box.GetMin(), aligned_box.GetMax()


def _print_corridor_metrics(stage) -> None:
    """
    测量并输出当前走廊的实际物理指标（包括真实净宽和墙体厚度）
    通过计算墙体之间的世界空间包围盒距离来获得精确数值
    """
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    unit_label = f"units (metersPerUnit={mpu})" if mpu else "units"

    # 第一段：两面墙沿 Y 方向的净宽
    min_a, max_a = _world_min_max(stage, f"{WORLD_ENV_PATH}/Wall_X_PosY")
    min_b, max_b = _world_min_max(stage, f"{WORLD_ENV_PATH}/Wall_X_NegY")
    clear_y = float(min_a[1] - max_b[1])  # 上墙内侧(minY) - 下墙内侧(maxY)
    thick_y_pos = float(max_a[1] - min_a[1])
    thick_y_neg = float(max_b[1] - min_b[1])

    # 第二段：两面墙沿 X 方向的净宽
    min_c, max_c = _world_min_max(stage, f"{WORLD_ENV_PATH}/Wall_Y_Inner")
    min_d, max_d = _world_min_max(stage, f"{WORLD_ENV_PATH}/Wall_Y_Outer")
    clear_x = float(min_d[0] - max_c[0])  # 右墙内侧(minX) - 左墙内侧(maxX)
    thick_x_inner = float(max_c[0] - min_c[0])
    thick_x_outer = float(max_d[0] - min_d[0])

    def to_m(v: float) -> float:
        return v * float(mpu) if mpu else v

    print(
        f"[Corridor]第一段(沿X) clearWidthY={clear_y:.3f} {unit_label} (≈{to_m(clear_y):.3f}m), "
        f"thicknessY(pos/neg)={thick_y_pos:.3f}/{thick_y_neg:.3f}"
    )
    print(
        f"[Corridor]第二段(沿Y) clearWidthX={clear_x:.3f} {unit_label} (≈{to_m(clear_x):.3f}m), "
        f"thicknessX(inner/outer)={thick_x_inner:.3f}/{thick_x_outer:.3f}"
    )

def create_wall(
    stage,
    mat_helper: MaterialHelper,
    name: str,
    pos: Vec3,
    scale: Vec3,
    color: Color3,
) -> UsdGeom.Cube:
    """
    在舞台上创建一个立方体形式的墙体，并设置其位置、缩放、物理碰撞属性和颜色
    """
    path = f"{WORLD_ENV_PATH}/{name}"
    wall = UsdGeom.Cube.Define(stage, path)
    wall.CreateSizeAttr(1.0)
    xform_api = UsdGeom.XformCommonAPI(wall.GetPrim())
    xform_api.SetScale(_as_vec3(scale))
    xform_api.SetTranslate(_as_vec3d(pos))
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    mat_helper.apply(wall, color, name_suffix=f"{name}_Material")
    return wall


def create_obstacle(stage, mat_helper: MaterialHelper, name: str, pos: Vec3, scale: Vec3, color: Color3) -> UsdGeom.Cube:
    """
    创建一个立方体障碍物，具有碰撞属性和自定义颜色
    """
    obs = UsdGeom.Cube.Define(stage, f"{WORLD_ENV_PATH}/{name}")
    obs.CreateSizeAttr(1.0)
    xform_api = UsdGeom.XformCommonAPI(obs.GetPrim())
    xform_api.SetScale(_as_vec3(scale))
    xform_api.SetTranslate(_as_vec3d(pos))
    mat_helper.apply(obs, color, name_suffix=f"{name}_Material")
    UsdPhysics.CollisionAPI.Apply(obs.GetPrim())
    return obs


def create_marker_sphere(stage, mat_helper: MaterialHelper, name: str, pos: Vec3, scale: Vec3, color: Color3) -> UsdGeom.Sphere:
    """
    创建一个球体标记（如起点或终点），通常不设为物理碰撞体
    """
    prim = UsdGeom.Sphere.Define(stage, f"{WORLD_ENV_PATH}/{name}")
    xform_api = UsdGeom.XformCommonAPI(prim.GetPrim())
    xform_api.SetScale(_as_vec3(scale))
    xform_api.SetTranslate(_as_vec3d(pos))
    mat_helper.apply(prim, color)
    return prim


def import_car(
    stage,
    usd_path: str,
    prim_path: str,
    pos: Vec3,
) -> Usd.Prim:
    car_prim = stage.DefinePrim(prim_path, "Xform")
    car_prim.GetReferences().AddReference(usd_path)
    xform_api = UsdGeom.XformCommonAPI(car_prim)
    xform_api.SetTranslate(_as_vec3d(pos))
    return car_prim


def clear_car(stage) -> None:
    """
    清理场景中的小车
    """
    _remove_if_exists(stage, CAR_PATH)


def build_environment(stage) -> None:
    """
    核心环境构建流程：包括清理旧场景、创建地面、构建 L 型走廊墙体、放置 500 个随机障碍物以及起始/目标标记点
    """
    # 清空旧环境（保证脚本可重复运行）
    _remove_if_exists(stage, WORLD_ENV_PATH)
    _remove_if_exists(stage, GROUND_PATH)
    clear_car(stage)

    UsdGeom.Xform.Define(stage, WORLD_ENV_PATH)

    # ===============================
    # 创建地面
    # ===============================
    GroundPlane(prim_path=GROUND_PATH, z_position=0)

    mat_helper = MaterialHelper(stage)

    # 以“净宽 + 墙厚”自动计算墙中心位置：
    # 这样你只改 WALL_THICKNESS，墙的内侧面位置不会漂移。
    corridor_half_width = (float(CORRIDOR_CLEAR_WIDTH) + float(WALL_THICKNESS)) / 2.0
    leg_x_center = float(TURN_X) - float(LEG_X_LENGTH) / 2.0
    leg_y_center = float(LEG_Y_LENGTH) / 2.0

    # ===============================
    # 走廊墙体（L 型 + 加宽）
    # ===============================
    # 第一段：沿 X 方向
    create_wall(
        stage,
        mat_helper,
        "Wall_X_PosY",
        pos=(leg_x_center, +corridor_half_width, 0.25),
        scale=(LEG_X_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
        color=(1, 0, 0),
    )
    create_wall(
        stage,
        mat_helper,
        "Wall_X_NegY",
        pos=(leg_x_center, -corridor_half_width, 0.25),
        scale=(LEG_X_LENGTH+(CORRIDOR_CLEAR_WIDTH+WALL_THICKNESS)*2, WALL_THICKNESS, WALL_HEIGHT),
        color=(0, 0, 1),
    )

    # 第二段：在 TURN_X 处向 +Y 转弯
    # 内侧墙从 y=+half_width 开始，避免切穿第一段的上侧空间
    create_wall(
        stage,
        mat_helper,
        "Wall_Y_Inner",
        pos=(9.5, 12, 0.5),
        scale=(WALL_THICKNESS, LEG_Y_LENGTH, WALL_HEIGHT),
        color=(1, 0, 0),
    )
    # 外侧墙从 y=-half_width 开始，与第一段下侧连贯
    create_wall(
        stage,
        mat_helper,
        "Wall_Y_Outer",
        pos=(12.5, 9, 0.25),
        scale=(WALL_THICKNESS, LEG_Y_LENGTH, WALL_HEIGHT),
        color=(0, 0, 1),
    )

    # ===============================
    # 输出：墙厚 / 中心距 / 净宽（内侧面间距）
    # ===============================
    # 说明：UsdGeom.Cube size=1.0 时，沿某轴的实际厚度 = scale 该轴值
    wall_thickness_axis = float(WALL_THICKNESS)
    centerline_distance = 2.0 * corridor_half_width
    clear_width = centerline_distance - wall_thickness_axis
    # 读取墙的实际世界包围盒，输出“真实净宽/真实厚度”（推荐以这个为准）
    _print_corridor_metrics(stage)

    # ===============================
    # 随机障碍物
    # ===============================
    for i in range(200):
        x = random.uniform(-60, 60)
        y = random.uniform(-60, 60)
        create_obstacle(
            stage,
            mat_helper,
            name=f"Obstacle_{i}",
            pos=(x, y, 0.25),
            scale=(0.5, 0.5, 0.5),
            color=(1, 1, 0),
        )

    # Spawn 红球
    create_marker_sphere(stage, mat_helper, name="Spawn", pos=(-4.5, 0, 0.3), scale=(0.3, 0.3, 0.3), color=(1.0, 0.0, 0.0))

    # Goal 绿球
    create_marker_sphere(stage, mat_helper, name="Goal", pos=(4.5, 0, 0.3), scale=(0.3, 0.3, 0.3), color=(0.0, 1.0, 0.0))

    # ===============================
    # 导入小车
    # ===============================
    # 建议 Z 轴设为 0.5 左右，确保小车在地面之上。
    # 默认开启传感器，也可以根据需要通过参数关闭。
    import_car(
        stage,
        CAR_USD_PATH,
        CAR_PATH,
        pos=(-4.5, 0, 0.5)
    )

    # ===============================
    # 加光源
    # ===============================
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path(LIGHT_PATH))
    distant_light.CreateIntensityAttr(300)


def main() -> None:
    """
    脚本入口函数：获取当前 USD 舞台并执行环境构建
    """
    stage = get_current_stage()

    build_environment(stage)
    print("✅ 场景生成完成")

if __name__ == "__main__":
    main()

