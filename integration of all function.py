#!/usr/bin/env python3
# coding: utf8

import rospy
import cv2
import numpy as np
import tf
import random
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from actionlib_msgs.msg import GoalID
from move_base_msgs.msg import MoveBaseActionResult
from jethexa_controller import client
from vision_utils import fps, get_area_max_contour, colors

class IntegratedExplorerNode:
    def __init__(self):
        rospy.init_node("integrated_explorer_node", anonymous=True)

        # --- 1. 初始化控制与导航参数 ---
        self.jethexa = client.Client(self)
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.explore_range = 2.0  # 随机导航范围（米）
        
        # --- 2. 视觉检测初始化 ---
        self.target_color_name = 'red'
        self.color_ranges = rospy.get_param('/lab_config_manager/color_range_list', None)
        self.target_color_range = self.color_ranges[self.target_color_name]
        self.fps = fps.FPS()
        self.red_detected = False

        # --- 3. 状态感知变量 ---
        self.current_pitch = 0.0
        self.current_gait = 1  # 默认 Tripod (1)
        self.pitch_threshold = 5.0  # 切换步态的角度阈值

        # --- 4. 订阅与发布 ---
        # 视觉
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        # 导航与定位
        self.goal_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        self.status_sub = rospy.Subscriber('move_base/result', MoveBaseActionResult, self.status_callback)
        # 传感器
        self.imu_sub = rospy.Subscriber("jethexa_controller/imu", Imu, self.imu_callback)

        rospy.loginfo("Integrated Explorer Node Initialized. Starting exploration...")
        rospy.sleep(2)
        self.publish_random_goal()

    # --- 步态切换逻辑 (基于IMU) ---
    def imu_callback(self, msg):
        orientation_q = msg.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        self.current_pitch = abs(pitch * 180.0 / 3.1415926)

        if not self.red_detected:
            # CPG算法简化逻辑：根据地形实时切换步态[cite: 10]
            if self.current_pitch > self.pitch_threshold and self.current_gait != 2:
                rospy.loginfo("Slope detected! Switching to Ripple Gait (Stability).")
                self.current_gait = 2
                self.update_locomotion()
            elif self.current_pitch <= self.pitch_threshold and self.current_gait != 1:
                rospy.loginfo("Flat ground detected! Switching to Tripod Gait (Speed).")
                self.current_gait = 1
                self.update_locomotion()

    def update_locomotion(self):
        # 向控制器发送最新的步态指令
        self.jethexa.traveling(gait=self.current_gait, stride=40.0, height=15.0, direction=0, time=1, steps=0)

    # --- 视觉检测逻辑 ---
    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        rgb_image = cv2.resize(rgb_image, (320, 180))
        
        try:
            img_blur = cv2.GaussianBlur(rgb_image, (3, 3), 3)
            img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
            mask = cv2.inRange(img_lab, tuple(self.target_color_range['min']), tuple(self.target_color_range['max']))
            
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            max_contour_area = get_area_max_contour(contours, 20)
            
            if max_contour_area is not None and not self.red_detected:
                self.red_detected = True
                self.stop_all_motion()
                    
        except Exception as e:
            rospy.logerr(f"Vision error: {e}")

    # --- 停止逻辑[cite: 9] ---
    def stop_all_motion(self):
        # 取消导航目标
        self.cancel_pub.publish(GoalID())
        # 强制停止底层控制器运动[cite: 10]
        self.jethexa.traveling(gait=0)
        rospy.logwarn("RED OBJECT DETECTED! ALL MOTIONS STOPPED.")

    # --- 自主探索逻辑[cite: 11] ---
    def publish_random_goal(self):
        if self.red_detected: return
        
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        # 随机生成探索点
        pose.pose.position.x = random.uniform(-self.explore_range, self.explore_range)
        pose.pose.position.y = random.uniform(-self.explore_range, self.explore_range)
        pose.pose.orientation.w = 1.0
        
        rospy.loginfo(f"Exploring new random goal: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}")
        self.goal_pub.publish(pose)
        # 启动初始步态
        self.update_locomotion()

    def status_callback(self, msg):
        # 状态3表示成功到达[cite: 11]
        if msg.status.status == 3:
            rospy.loginfo("Goal reached. Planning next exploration point...")
            rospy.sleep(1.0)
            self.publish_random_goal()
        elif msg.status.status in [4, 5, 9]: # 目标无法到达或取消
            if not self.red_detected:
                rospy.logwarn("Goal failed, retrying another point...")
                self.publish_random_goal()

if __name__ == "__main__":
    try:
        node = IntegratedExplorerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
