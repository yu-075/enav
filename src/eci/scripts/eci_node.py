#!/usr/bin/env python3
"""ECI 节点：从点云计算环境拥挤指标并发布。

处理链路：
1) 订阅输入 PointCloud2（默认 /sim/point_cloud）；
2) 转成 XYZ numpy 数组；
3) 体素降采样；
4) 高度、前向视场、距离过滤；
5) 计算 density 与 width_metric；
6) 按加权公式得到 ECI，并以 5Hz 发布到 /eci_value。
7) 发布检测范围 Marker 到 RViz。
"""

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class EciNode(Node):
    """ECI 计算节点。

    - 输入：PointCloud2（参数 input_topic）
    - 输出：Float32（参数 output_topic）
    """

    def __init__(self) -> None:
        super().__init__('eci_node')

        # 可调参数：降采样精度、密度归一化上限点数、前向最大距离。
        self.declare_parameter('input_topic', '/sim/point_cloud')
        self.declare_parameter('output_topic', '/eci_value')
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('max_points', 3000.0)
        self.declare_parameter('max_distance', 5.0)
        self.declare_parameter('speed_topic', '/odom')
        self.declare_parameter('prediction_time_sec', 2.0)
        self.declare_parameter('zero_speed_distance', 3.0)
        self.declare_parameter('min_height', 0.05)
        self.declare_parameter('max_height', 1.0)
        self.declare_parameter('density_distance', 3.0)
        self.declare_parameter('density_half_fov_deg', 45.0)
        self.declare_parameter('width_half_fov_deg', 30.0)
        self.declare_parameter('no_cloud_reset_sec', 8.0)
        self.declare_parameter('eci_publish_hz', 20.0)
        self.declare_parameter('eci_smoothing_alpha', 0.6)
        self.declare_parameter('eci_zero_epsilon', 1e-6)
        self.declare_parameter('enable_range_visualization', True)
        self.declare_parameter('range_marker_topic', '/eci_detection_range')
        self.declare_parameter('range_target_frame', '/Leatherback')

        # 话题可通过 YAML 统一配置。
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.max_points = float(self.get_parameter('max_points').value)
        self.max_distance = float(self.get_parameter('max_distance').value)
        self.speed_topic = str(self.get_parameter('speed_topic').value)
        self.prediction_time_sec = float(self.get_parameter('prediction_time_sec').value)
        self.zero_speed_distance = float(self.get_parameter('zero_speed_distance').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.density_distance = float(self.get_parameter('density_distance').value)
        self.density_half_fov = math.radians(
            float(self.get_parameter('density_half_fov_deg').value)
        )
        self.width_half_fov = math.radians(
            float(self.get_parameter('width_half_fov_deg').value)
        )
        self.no_cloud_reset_sec = float(self.get_parameter('no_cloud_reset_sec').value)
        self.eci_publish_hz = float(self.get_parameter('eci_publish_hz').value)
        self.eci_smoothing_alpha = float(self.get_parameter('eci_smoothing_alpha').value)
        self.eci_smoothing_alpha = float(np.clip(self.eci_smoothing_alpha, 0.0, 1.0))
        self.eci_zero_epsilon = float(self.get_parameter('eci_zero_epsilon').value)
        self.eci_zero_epsilon = max(self.eci_zero_epsilon, 0.0)
        self.enable_range_visualization = bool(
            self.get_parameter('enable_range_visualization').value
        )
        self.range_marker_topic = str(self.get_parameter('range_marker_topic').value)
        self.range_target_frame = str(self.get_parameter('range_target_frame').value).lstrip('/')

        # 最新计算结果缓存；由回调更新，由定时器发布。
        self.latest_eci = 0.0
        self.has_eci_history = False
        self.latest_density = 0.0
        self.latest_width_metric = 0.0
        self.last_cloud_stamp = None
        self.visualization_source_frame = 'odom'
        self.current_speed = 0.0
        self.current_detection_distance = max(self.zero_speed_distance, 0.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_tf_warn_ns = 0
        self.last_height_filter_warn_ns = 0

        # 点云到来时执行计算。
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10,
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.speed_topic,
            self.odom_callback,
            10,
        )
        # 结果发布器与定时器。
        self.publisher = self.create_publisher(Float32, self.output_topic, 10)
        self.range_marker_publisher = self.create_publisher(
            MarkerArray,
            self.range_marker_topic,
            10,
        )
        publish_period_sec = 1.0 / max(self.eci_publish_hz, 1.0)
        self.timer = self.create_timer(publish_period_sec, self.publish_eci)

        self.get_logger().info(
            f'eci_node started: subscribe={self.input_topic}, publish={self.output_topic}, '
            f'voxel_size={self.voxel_size}, max_points={self.max_points}, max_distance={self.max_distance}, '
            f'speed_topic={self.speed_topic}, prediction_time_sec={self.prediction_time_sec}, '
            f'zero_speed_distance={self.zero_speed_distance}, '
            f'min_height={self.min_height}, max_height={self.max_height}, '
            f'density_distance={self.density_distance}, '
            f'density_half_fov_deg={math.degrees(self.density_half_fov)}, '
            f'width_half_fov_deg={math.degrees(self.width_half_fov)}, '
            f'no_cloud_reset_sec={self.no_cloud_reset_sec}, '
            f'eci_publish_hz={self.eci_publish_hz}, '
            f'eci_smoothing_alpha={self.eci_smoothing_alpha}, '
            f'eci_zero_epsilon={self.eci_zero_epsilon}, '
            f'enable_range_visualization={self.enable_range_visualization}, '
            f'range_marker_topic={self.range_marker_topic}, '
            f'range_target_frame={self.range_target_frame}'
        )

    def odom_callback(self, msg: Odometry) -> None:
        """里程计回调：更新当前速度与动态检测距离。"""
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.current_speed = float(math.hypot(vx, vy))

        # 检测距离 = max(静止基线距离, 当前车速在 prediction_time_sec 内的位移)。
        self.current_detection_distance = float(
            max(
                max(self.zero_speed_distance, 0.0),
                self.current_speed * max(self.prediction_time_sec, 0.0),
            )
        )

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        """点云回调：完成预处理与 ECI 计算。"""
        self.last_cloud_stamp = self.get_clock().now()
        if msg.header.frame_id:
            self.visualization_source_frame = msg.header.frame_id.lstrip('/')
        points = self.pointcloud2_to_xyz(msg)
        if points.size == 0:
            # 无点云时输出 0，避免发布陈旧值。
            self.update_smoothed_eci(0.0)
            self.latest_density = 0.0
            self.latest_width_metric = 0.0
            return

        # 1) 体素降采样，2) 规则过滤。
        downsampled = self.voxel_downsample(points, self.voxel_size)
        filtered = self.apply_filters(downsampled, self.current_detection_distance)

        # 密度：按“前方 3m 且 ±45°”内的点数量归一化。
        density = self.compute_density_metric(filtered)

        # 宽度：按（最长连续空闲角段反比）计算。
        width_metric = self.compute_width_metric(filtered, self.current_detection_distance)

        # ECI 加权融合。
        eci = float(np.clip(0.6 * density + 0.4 * width_metric, 0.0, 1.0))

        self.latest_density = density
        self.latest_width_metric = width_metric
        self.update_smoothed_eci(eci)

    def update_smoothed_eci(self, current_eci: float) -> None:
        """按指数滑动平均更新 ECI，降低帧间抖动。"""
        if not self.has_eci_history:
            self.latest_eci = float(np.clip(current_eci, 0.0, 1.0))
            self.has_eci_history = True
            return

        alpha = self.eci_smoothing_alpha
        smoothed = alpha * float(current_eci) + (1.0 - alpha) * float(self.latest_eci)
        self.latest_eci = float(np.clip(smoothed, 0.0, 1.0))

        # 抑制指数平滑的数值拖尾，避免长期输出 1e-12 这类非物理微小值。
        if abs(self.latest_eci) < self.eci_zero_epsilon:
            self.latest_eci = 0.0

    def publish_eci(self) -> None:
        """定时发布最新 ECI。"""
        msg = Float32()
        now = self.get_clock().now()
        is_long_time_no_cloud = (
            self.last_cloud_stamp is None
            or (now - self.last_cloud_stamp).nanoseconds
            > int(max(self.no_cloud_reset_sec, 0.0) * 1e9)
        )

        # 仅在“持续较长时间完全没点云”时归零；
        # 短时断流保留上一帧，避免数值在 0 与正常值之间抖动。
        if is_long_time_no_cloud:
            self.latest_density = 0.0
            self.latest_width_metric = 0.0
            self.latest_eci = 0.0
            self.has_eci_history = False
        
        msg.data = float(self.latest_eci)
        self.publisher.publish(msg)
        self.publish_detection_range_markers()
        self.get_logger().info(
            f'eci: {msg.data} ')

    def publish_detection_range_markers(self) -> None:
        """发布 RViz 检测范围可视化。"""
        if not self.enable_range_visualization:
            return

        source_frame = self.visualization_source_frame
        target_frame = self.range_target_frame

        marker_array = MarkerArray()
        marker_array.markers.append(
            self.make_sector_marker(
                marker_id=0,
                radius=self.density_distance,
                half_fov=self.density_half_fov,
                color_rgba=(1.0, 0.2, 0.2, 0.9),
                source_frame=source_frame,
                target_frame=target_frame,
            )
        )
        marker_array.markers.append(
            self.make_sector_marker(
                marker_id=1,
                radius=self.current_detection_distance,
                half_fov=self.width_half_fov,
                color_rgba=(0.2, 0.9, 1.0, 0.9),
                source_frame=source_frame,
                target_frame=target_frame,
            )
        )
        self.range_marker_publisher.publish(marker_array)

    def make_sector_marker(
        self,
        marker_id: int,
        radius: float,
        half_fov: float,
        color_rgba: tuple[float, float, float, float],
        source_frame: str,
        target_frame: str,
    ) -> Marker:
        """构建扇区轮廓线 Marker。"""
        marker = Marker()
        marker.header.frame_id = target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'eci_detection_range'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        marker.frame_locked = True

        point_count = 60
        angles = np.linspace(-half_fov, half_fov, point_count)

        local_points = [[0.0, 0.0, 0.0]]
        for angle in angles:
            local_points.append(
                [
                    float(radius * math.cos(float(angle))),
                    float(radius * math.sin(float(angle))),
                    0.0,
                ]
            )
        local_points.append([0.0, 0.0, 0.0])

        transformed_points = self.transform_points(local_points, source_frame, target_frame)
        if transformed_points is None:
            # TF 不可用时回退到源坐标系，至少保证可视化不断流。
            marker.header.frame_id = source_frame
            transformed_points = local_points

        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in transformed_points]
        return marker

    def transform_points(
        self,
        points: list[list[float]],
        source_frame: str,
        target_frame: str,
    ) -> Optional[list[list[float]]]:
        """将点集从 source_frame 变换到 target_frame。"""
        if source_frame == target_frame:
            return points

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_tf_warn_ns > int(1e9):
                self.get_logger().warn(
                    f'Cannot transform detection marker from {source_frame} to {target_frame}'
                )
                self.last_tf_warn_ns = now_ns
            return None

        tx = float(transform.transform.translation.x)
        ty = float(transform.transform.translation.y)
        tz = float(transform.transform.translation.z)
        qx = float(transform.transform.rotation.x)
        qy = float(transform.transform.rotation.y)
        qz = float(transform.transform.rotation.z)
        qw = float(transform.transform.rotation.w)

        rot = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
        transformed_points = []
        for point in points:
            vec = np.array(point, dtype=np.float64)
            rotated = rot @ vec
            transformed_points.append(
                [
                    float(rotated[0] + tx),
                    float(rotated[1] + ty),
                    float(rotated[2] + tz),
                ]
            )
        return transformed_points

    @staticmethod
    def quaternion_to_rotation_matrix(
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        """四元数转旋转矩阵。"""
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return np.eye(3)
        x = qx / norm
        y = qy / norm
        z = qz / norm
        w = qw / norm

        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def compute_density_metric(self, points: np.ndarray) -> float:
        """计算前方扇区内带二次距离衰减的障碍密度，输出范围 [0, 1]。"""
        if points.size == 0:
            return 0.0

        angles = np.arctan2(points[:, 1], points[:, 0])
        distances = np.linalg.norm(points[:, :2], axis=1)
        detection_distance = max(self.density_distance, 1e-6)

        density_mask = (
            (angles >= -self.density_half_fov)
            & (angles <= self.density_half_fov)
            & (distances < detection_distance)
        )

        if not np.any(density_mask):
            return 0.0

        # 单点权重：((检测距离 - 点距) / 检测距离)^2，近点贡献更高。
        masked_distances = distances[density_mask]
        point_weights = np.square((detection_distance - masked_distances) / detection_distance)
        weighted_density = float(np.sum(point_weights))
        density = float(np.clip(weighted_density / max(self.max_points, 1.0), 0.0, 1.0))
        return density

    @staticmethod
    def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
        """将 PointCloud2 转为 Nx3 的 float32 数组（XYZ）。"""
        xyz_iter = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )
        xyz_list = []
        # 兼容不同 ROS 发行版/实现下 read_points 的返回类型：
        # - 结构化点（np.void）
        # - 普通 tuple/list
        for point in xyz_iter:
            if isinstance(point, np.void):
                xyz_list.append([float(point['x']), float(point['y']), float(point['z'])])
            else:
                xyz_list.append([float(point[0]), float(point[1]), float(point[2])])
        if not xyz_list:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(xyz_list, dtype=np.float32)

    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """体素降采样：每个体素保留一个点。"""
        if points.size == 0:
            return points
        if voxel_size <= 0.0:
            return points

        # 将连续坐标映射到体素网格索引，再按唯一索引去重。
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        return points[unique_indices]

    def apply_filters(self, points: np.ndarray, detection_distance: float) -> np.ndarray:
        """按规则过滤点云：高度、前向角、距离。"""
        if points.size == 0:
            return points

        # 高度过滤：默认去掉贴地回波，降低空旷场景误判。
        z = points[:, 2]
        z_mask = (z > self.min_height) & (z < self.max_height)
        p = points[z_mask]
        if p.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        x = p[:, 0]
        y = p[:, 1]
        angles = np.arctan2(y, x)
        distances = np.linalg.norm(p[:, :2], axis=1)
        max_distance = max(float(detection_distance), 0.0)

        if max_distance == 0.0:
            return np.empty((0, 3), dtype=np.float32)

        # 前向区域过滤：-90° < atan2(y, x) < 90°，且距离 < 动态检测距离。
        front_mask = (
            (angles > -math.pi / 2.0)
            & (angles < math.pi / 2.0)
            & (distances < max_distance)
        )
        p = p[front_mask]

        if p.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return p.astype(np.float32, copy=False)

    def compute_width_metric(self, points: np.ndarray, detection_distance: float) -> float:
        """计算简单“自由空间宽度”指标。

        方法：
        - 仅看前方 ±30° 扇区；
        - 将角度离散为若干 bin，含点即视为占用；
        - 找到最长连续“空闲 bin”长度并归一化为 free_ratio；
        - width_metric = 1 - free_ratio。

        说明：该定义下，值越大表示越拥挤（可通行宽度越小）。
        """
        if points.size == 0:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x)
        distances = np.linalg.norm(points[:, :2], axis=1)
        max_distance = max(float(detection_distance), 0.0)

        if max_distance == 0.0:
            return 0.0

        sector_limit = self.width_half_fov
        sector_mask = (
            (angles >= -sector_limit)
            & (angles <= sector_limit)
            & (distances < max_distance)
        )
        sector_angles = angles[sector_mask]

        # 角向离散精度：bin 越多，角向分辨率越细。
        bin_count = 90
        if sector_angles.size == 0:
            # 扇区内无障碍，占用度最低。
            return 0.0

        edges = np.linspace(-sector_limit, sector_limit, bin_count + 1)
        indices = np.digitize(sector_angles, edges) - 1
        indices = np.clip(indices, 0, bin_count - 1)

        occupied = np.zeros(bin_count, dtype=bool)
        occupied[indices] = True
        free = ~occupied

        # 统计最长连续空闲段长度。
        max_free_run = 0
        current_run = 0
        for is_free in free:
            if is_free:
                current_run += 1
                if current_run > max_free_run:
                    max_free_run = current_run
            else:
                current_run = 0

        free_ratio = float(max_free_run) / float(bin_count)
        width_metric = float(np.clip(1.0 - free_ratio, 0.0, 1.0))
        return width_metric


def main(args: Optional[list[str]] = None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = EciNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
