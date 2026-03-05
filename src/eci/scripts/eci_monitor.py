#!/usr/bin/env python3
"""ECI 可视化监视器：实时显示数值与动态曲线。"""

from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class EciMonitorNode(Node):
    """订阅 /eci_value，并缓存时间序列用于可视化。"""

    def __init__(self) -> None:
        super().__init__('eci_monitor')

        self.declare_parameter('topic', '/eci_value')
        self.declare_parameter('window_seconds', 30.0)

        self.topic = str(self.get_parameter('topic').value)
        self.window_seconds = float(self.get_parameter('window_seconds').value)

        self.latest_value = 0.0
        self.latest_stamp = None
        self.start_time = self.get_clock().now()

        self.time_buffer: deque[float] = deque()
        self.value_buffer: deque[float] = deque()

        self.subscription = self.create_subscription(
            Float32,
            self.topic,
            self.eci_callback,
            10,
        )

        self.get_logger().info(
            f'eci_monitor started: subscribe={self.topic}, window_seconds={self.window_seconds}'
        )

    def eci_callback(self, msg: Float32) -> None:
        """接收 ECI 并保存到缓冲区。"""
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        value = float(msg.data)
        self.latest_value = value
        self.latest_stamp = t

        self.time_buffer.append(t)
        self.value_buffer.append(value)

        threshold = t - self.window_seconds
        while self.time_buffer and self.time_buffer[0] < threshold:
            self.time_buffer.popleft()
            self.value_buffer.popleft()


def main(args: Optional[list[str]] = None) -> None:
    """节点入口：创建 matplotlib 实时窗口。"""
    rclpy.init(args=args)
    node = EciMonitorNode()

    selected_style = None
    for style_name in ('seaborn-v0_8', 'seaborn', 'ggplot', 'default'):
        try:
            plt.style.use(style_name)
            selected_style = style_name
            break
        except OSError:
            continue
    if selected_style is None:
        plt.style.use('default')
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

    ax_info = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[1])

    ax_info.axis('off')
    value_text = ax_info.text(
        0.02,
        0.68,
        'ECI: 0.000',
        fontsize=28,
        fontweight='bold',
        transform=ax_info.transAxes,
    )
    stats_text = ax_info.text(
        0.02,
        0.22,
        'min=0.000  max=0.000  mean=0.000',
        fontsize=14,
        transform=ax_info.transAxes,
    )

    ax_plot.set_title('ECI Dynamic Trend')
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('ECI')
    ax_plot.set_ylim(0.0, 1.0)
    ax_plot.grid(True, alpha=0.3)

    line, = ax_plot.plot([], [], linewidth=2.2)

    def update(_: int):
        rclpy.spin_once(node, timeout_sec=0.0)

        if not node.time_buffer:
            return line, value_text, stats_text

        times = list(node.time_buffer)
        values = list(node.value_buffer)

        line.set_data(times, values)

        x_min = times[0]
        x_max = max(times[-1], x_min + 1e-3)
        ax_plot.set_xlim(x_min, x_max)

        current = values[-1]
        v_min = min(values)
        v_max = max(values)
        v_mean = sum(values) / len(values)

        value_text.set_text(f'ECI: {current:.3f}')
        stats_text.set_text(f'min={v_min:.3f}  max={v_max:.3f}  mean={v_mean:.3f}')
        return line, value_text, stats_text

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    _ = anim

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
