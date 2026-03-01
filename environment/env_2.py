import sys
import os
from typing import Iterable, Sequence, Tuple
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, UsdLux
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.core.api.objects.ground_plane import GroundPlane

# 获取当前脚本所在目录并加入 sys.path
env_path = "/root/enav/environment"
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
from env_1 import build_nav_environment

def build_env2() -> None:
    stage = get_current_stage()
    # 1. 调用 env_1 的导航环境 (包含边界和基本分割)
    build_nav_environment(stage)
    # 2. 在 main 中额外直接创建一些墙体
    mat_helper = MaterialHelper(stage)
    z_height = WALL_HEIGHT / 2.0

    # 使用统一的 WALL_COLOR
    color = WALL_COLOR

    # --- 象限1 (+X, +Y) 的额外墙体 ---
    create_wall(stage, mat_helper, "Ex_Q1_V", (8, 8, z_height), (WALL_THICKNESS, 5, WALL_HEIGHT), color)
    create_wall(stage, mat_helper, "Ex_Q1_H", (10, 6, z_height), (5, WALL_THICKNESS, WALL_HEIGHT), color)

    # --- 象限2 (-X, +Y) 的额外墙体 ---
    create_wall(stage, mat_helper, "Ex_Q2_V", (-8, 7, z_height), (WALL_THICKNESS, 6, WALL_HEIGHT), color)
    create_wall(stage, mat_helper, "Ex_Q2_H", (-6, 9, z_height), (7, WALL_THICKNESS, WALL_HEIGHT), color)

    # --- 象限3 (-X, -Y) 的额外墙体 ---
    # 创建一个狭窄的弯道或 U 型区域
    create_wall(stage, mat_helper, "Ex_Q3_U1", (-10, -6, z_height), (WALL_THICKNESS, 6, WALL_HEIGHT), color)
    create_wall(stage, mat_helper, "Ex_Q3_U2", (-8, -9, z_height), (6, WALL_THICKNESS, WALL_HEIGHT), color)
    create_wall(stage, mat_helper, "Ex_Q3_U3", (-6, -6, z_height), (WALL_THICKNESS, 6, WALL_HEIGHT), color)

    # --- 象限4 (+X, -Y) 的额外墙体 ---
    # 添加几根“隔离柱”
    for i in range(3):
        create_wall(stage, mat_helper, f"Ex_Q4_Pillar_{i}", (4 + i*3, -7, z_height), (1.5, 1.5, WALL_HEIGHT), color)

    # --- 中心区域补充 ---
    create_wall(stage, mat_helper, "Ex_Center_Diagonal", (0, -4, z_height), (6, WALL_THICKNESS, WALL_HEIGHT), color)

    print(f"✅ 已基于 env_1 场景，使用 WALL_COLOR 额外创建了更多导航墙体，场景现已更加复杂。")

if __name__ == "__main__":
    build_env2()
