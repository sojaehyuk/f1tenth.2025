import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
# QoS 설정을 위해 다음 클래스들을 import 합니다.
from rclpy.qos import QoSProfile, DurabilityPolicy
import random
import math
import numpy as np

class ObstacleAndLidarSimulator(Node):
    def __init__(self):
        super().__init__('obstacle_lidar_simulator')

        # --- 설정값 ---
        self.MAP_FRAME = 'map'
        self.OBSTACLE_RADIUS_MIN = 0.2
        self.OBSTACLE_RADIUS_MAX = 0.5
        self.OBSTACLE_HEIGHT = 1.0
        # 📍 10초마다 장애물 씬을 업데이트하도록 설정
        self.OBSTACLE_UPDATE_INTERVAL = 10.0

        self.OBSTACLE_ZONES = [
            {'x_min': 1.9 , 'x_max': 4.0, 'y_min': 0.9, 'y_max': 3.0, 'count': 3},
            {'x_min': -9.1, 'x_max': -6.0, 'y_min': -0.9, 'y_max': 2.5, 'count': 3},
            #{'x_min': -3.0, 'x_max': 3.0, 'y_min': 5.0, 'y_max': 6.0, 'count': 5}
        ]
        
        # --- 가상 라이다 센서 설정 ---
        self.LIDAR_FRAME = 'laser_frame'
        self.LIDAR_TOPIC = '/scan'
        self.LIDAR_RATE = 10.0
        self.LIDAR_ANGLE_MIN = -math.pi
        self.LIDAR_ANGLE_MAX = math.pi
        self.LIDAR_NUM_READINGS = 360
        self.LIDAR_RANGE_MIN = 0.1
        self.LIDAR_RANGE_MAX = 10.0

        self.obstacles = []

        # --- QoS 설정 ---
        # RViz 같은 GUI 도구가 나중에 실행되어도 마지막 MarkerArray 메시지를 받을 수 있도록 설정
        marker_qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # --- 퍼블리셔 ---
        self.marker_publisher_ = self.create_publisher(
            MarkerArray, 
            '/visualization_marker_array', 
            marker_qos_profile) # 수정된 QoS 프로파일 적용
        self.scan_publisher_ = self.create_publisher(LaserScan, self.LIDAR_TOPIC, 10)
        
        # --- 타이머 분리 ---
        # 1. 장애물 씬 업데이트 타이머 (10초마다 update_scene 함수 호출)
        self.obstacle_timer = self.create_timer(
            self.OBSTACLE_UPDATE_INTERVAL, 
            self.update_scene)
            
        # 2. 라이다 발행 타이머 (기존처럼 빠른 주기로 publish_laserscan 함수 호출)
        self.lidar_timer = self.create_timer(
            1.0 / self.LIDAR_RATE, 
            self.publish_laserscan)

        self.angle_increment = (self.LIDAR_ANGLE_MAX - self.LIDAR_ANGLE_MIN) / self.LIDAR_NUM_READINGS
        
        # --- 초기 장애물 생성 ---
        self.get_logger().info('Generating initial obstacle scene.')
        self.update_scene()

    def update_scene(self):
        """10초마다 호출되어 장애물을 새로 생성하고 마커를 발행합니다."""
        self.get_logger().info(f'Updating obstacle scene (next update in {self.OBSTACLE_UPDATE_INTERVAL}s)...')
        # 1. 새로운 장애물 위치 데이터 생성
        self.generate_obstacles_in_zones()
        # 2. RViz에 마커 발행 (내부적으로 이전 마커를 모두 지우고 새로 그림)
        self.publish_obstacle_markers()

    def generate_obstacles_in_zones(self):
        self.obstacles.clear()
        for zone in self.OBSTACLE_ZONES:
            for _ in range(zone['count']):
                x = random.uniform(zone['x_min'], zone['x_max'])
                y = random.uniform(zone['y_min'], zone['y_max'])
                radius = random.uniform(self.OBSTACLE_RADIUS_MIN, self.OBSTACLE_RADIUS_MAX)
                self.obstacles.append({'x': x, 'y': y, 'radius': radius})

    def publish_obstacle_markers(self):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for i, obs in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = self.MAP_FRAME
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = obs['x']
            marker.pose.position.y = obs['y']
            marker.pose.position.z = self.OBSTACLE_HEIGHT / 2.0
            marker.pose.orientation.w = 1.0
            radius = obs['radius']
            marker.scale.x = radius * 2
            marker.scale.y = radius * 2
            marker.scale.z = self.OBSTACLE_HEIGHT
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 1.0;
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)
        
        self.marker_publisher_.publish(marker_array)
        self.get_logger().info(f'Published {len(self.obstacles)} obstacle markers from defined zones.')

    def publish_laserscan(self):
        # 라이다 스캔을 생성할 장애물이 없으면 아무것도 하지 않음
        if not self.obstacles:
            self.get_logger().warn('No obstacles to scan.', throttle_duration_sec=10)
            return

        now = self.get_clock().now()
        scan = LaserScan()
        # ... (이하 라이다 데이터 생성 로직은 기존과 동일) ...
        scan.header.stamp = now.to_msg()
        scan.header.frame_id = self.LIDAR_FRAME
        scan.angle_min = self.LIDAR_ANGLE_MIN
        scan.angle_max = self.LIDAR_ANGLE_MAX
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.LIDAR_RATE
        scan.range_min = self.LIDAR_RANGE_MIN
        scan.range_max = self.LIDAR_RANGE_MAX

        ranges = [self.LIDAR_RANGE_MAX] * self.LIDAR_NUM_READINGS

        for i in range(self.LIDAR_NUM_READINGS):
            angle = self.LIDAR_ANGLE_MIN + i * self.angle_increment
            min_dist_for_this_angle = self.LIDAR_RANGE_MAX
            
            for obs in self.obstacles:
                dx, dy = obs['x'], obs['y']
                a = 1.0
                b = -2 * (dx * math.cos(angle) + dy * math.sin(angle))
                c = dx**2 + dy**2 - obs['radius']**2
                discriminant = b**2 - 4*a*c
                
                if discriminant >= 0:
                    sqrt_discriminant = math.sqrt(discriminant)
                    t1 = (-b - sqrt_discriminant) / (2 * a)
                    if t1 > 0 and t1 < min_dist_for_this_angle:
                        min_dist_for_this_angle = t1
            ranges[i] = float(min_dist_for_this_angle)

        scan.ranges = ranges
        self.scan_publisher_.publish(scan)
        # 로그가 너무 많이 찍히는 것을 방지하기 위해 이 부분은 주석 처리하거나 삭제
        # self.get_logger().info(f'Published LaserScan with {len(ranges)} points.')


def main(args=None):
    rclpy.init(args=args)
    simulator_node = ObstacleAndLidarSimulator()
    rclpy.spin(simulator_node)
    simulator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
