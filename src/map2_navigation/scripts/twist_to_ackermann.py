#!/usr/bin/env python3

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TwistToAckermann(Node):
    def __init__(self) -> None:
        super().__init__("twist_to_ackermann")

        self.declare_parameter("wheelbase", 0.17)
        self.declare_parameter("max_steering_angle", 0.60)
        self.declare_parameter("min_speed", 0.05)
        self.declare_parameter("min_speed_for_steering", 0.01)
        self.declare_parameter("frame_id", "base_link")

        self._wheelbase = float(self.get_parameter("wheelbase").value)
        self._max_steering_angle = float(
            self.get_parameter("max_steering_angle").value
        )
        self._min_speed = float(self.get_parameter("min_speed").value)
        self._min_speed_for_steering = float(
            self.get_parameter("min_speed_for_steering").value
        )
        self._frame_id = str(self.get_parameter("frame_id").value)

        self._sub = self.create_subscription(Twist, "cmd_vel", self._on_twist, 10)
        self._pub = self.create_publisher(AckermannDriveStamped, "ackermann_cmd", 10)

    def _on_twist(self, msg: Twist) -> None:
        v = float(msg.linear.x)
        omega = float(msg.angular.z)

        # Ackermann cannot do true in-place rotation. If v is ~0 but omega != 0,
        # approximate by commanding a small speed with max steering.
        if abs(v) < self._min_speed_for_steering and abs(omega) > 1e-6:
            v = math.copysign(self._min_speed, omega)
            steering_angle = math.copysign(self._max_steering_angle, omega)
        elif abs(v) < 1e-6:
            v = 0.0
            steering_angle = 0.0
        else:
            steering_angle = math.atan(self._wheelbase * omega / v)
            steering_angle = max(
                -self._max_steering_angle, min(self._max_steering_angle, steering_angle)
            )

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.drive.speed = v
        out.drive.steering_angle = steering_angle

        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = TwistToAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
