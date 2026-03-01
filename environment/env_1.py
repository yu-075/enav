import sys
import os
from typing import Iterable, Sequence, Tuple
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, UsdLux
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.core.api.objects.ground_plane import GroundPlane

# 获取当前脚本所在目录并加入 sys.path
env_path = "/root/Desktop/environment"
if env_path not in sys.path:
    sys.path.append(env_path)

# 从 Corridor 导入现有的类和工具函数
from Corridor import (
    MaterialHelper, 
    create_wall, 
    _remove_if_exists, 
    WORLD_ENV_PATH, 
    GROUND_PATH, 
    LIGHT_PATH,
    WALL_THICKNESS,
    WALL_HEIGHT,
    build_square_environment,
    SPACE_SIZE,
    CAR_PATH,
    CAR_USD_PATH,
    WALL_COLOR,
    import_car,
    clear_car
)

def build_nav_environment(stage) -> None:
    # 1. 清理并初始化环境
    _remove_if_exists(stage, WORLD_ENV_PATH)
    _remove_if_exists(stage, GROUND_PATH)
    clear_car(stage)
    import_car(
        stage,
        CAR_USD_PATH,
        CAR_PATH,
        pos=(-4.5, 0, 0.5)
    )
    UsdGeom.Xform.Define(stage, WORLD_ENV_PATH)
    GroundPlane(prim_path=GROUND_PATH, z_position=0)
    mat_helper = MaterialHelper(stage)
    
    half_size = SPACE_SIZE / 2.0
    z_height = WALL_HEIGHT / 2.0
    build_square_environment(stage)
    # 3. 手动创建不规则分割墙 (Manual Irregular Walls)
    # 直接调用 create_wall，方便控制位置和长度，形成不完全封闭的区域
    # 中心十字形分割（带缺口）
    create_wall(stage, mat_helper, "Int_V1", (4, 7, z_height), (WALL_THICKNESS, 10, WALL_HEIGHT), WALL_COLOR)
    create_wall(stage, mat_helper, "Int_V2", (4, -7, z_height), (WALL_THICKNESS, 8, WALL_HEIGHT), WALL_COLOR)
    
    create_wall(stage, mat_helper, "Int_H1", (-7, 3, z_height), (12, WALL_THICKNESS, WALL_HEIGHT), WALL_COLOR)
    create_wall(stage, mat_helper, "Int_H2", (7, 3, z_height), (10, WALL_THICKNESS, WALL_HEIGHT), WALL_COLOR)

    # 一个不完全封闭的角落房间
    create_wall(stage, mat_helper, "Room_Corner_V", (-9, -6, z_height), (WALL_THICKNESS, 8, WALL_HEIGHT), WALL_COLOR)
    create_wall(stage, mat_helper, "Room_Corner_H", (-9, -2, z_height), (6, WALL_THICKNESS, WALL_HEIGHT), WALL_COLOR)

    # 另外一个独立墙段
    create_wall(stage, mat_helper, "Island_Wall", (8, -8, z_height), (6, WALL_THICKNESS, WALL_HEIGHT), WALL_COLOR)

    # 4. 灯光
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path(LIGHT_PATH))
    distant_light.CreateIntensityAttr(800)

def main() -> None:
    stage = get_current_stage()
    build_nav_environment(stage)
    print(f"✅ 已完成简化的不规则导航地图构建 (范围={SPACE_SIZE}x{SPACE_SIZE})")

if __name__ == "__main__":
    main()
