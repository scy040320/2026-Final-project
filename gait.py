#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from jethexa_controller import client
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

class MovingNode:
    def __init__(self):
        rospy.init_node("moving_node", anonymous=True, log_level=rospy.INFO)
        self.jethexa = client.Client(self)
        
        # 存储当前状态的变量
        self.current_velocity = 0.0
        self.current_pitch = 0.0
        self.start_pitch = None  # 用于记录初始姿态，计算变化量

        # 订阅里程计信息获取速度 (Odometry)
        self.odom_sub = rospy.Subscriber("jethexa_controller/odom", Odometry, self.odom_callback)
        
        # 订阅 IMU 信息获取俯仰角 (Pitch)
        self.imu_sub = rospy.Subscriber("jethexa_controller/imu", Imu, self.imu_callback)

    def odom_callback(self, msg):
        """里程计回调函数：获取实时速度"""
        # 获取线速度 (m/s)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        # 计算合成速度
        self.current_velocity = (vx**2 + vy**2)**0.5

    def imu_callback(self, msg):
        """IMU回调函数：获取实时俯仰角"""
        # 将四元数转换为欧拉角 (Euler angles)
        orientation_q = msg.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        )
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        
        # 弧度转角度
        self.current_pitch = pitch * 180.0 / 3.1415926
        
        if self.start_pitch is None:
            self.start_pitch = self.current_pitch

    def get_realtime_stats(self):
        """打印当前的实验数据"""
        pitch_change = self.current_pitch - (self.start_pitch if self.start_pitch else 0)
        rospy.loginfo("实时速度: {:.2f} m/s | 当前俯仰角: {:.2f}° | 俯仰角变化量: {:.2f}°".format(
            self.current_velocity, self.current_pitch, pitch_change))

    def forward(self, gait_type=1):
        """
        执行前进指令
        gait_type: 1 为 Ripple (波纹步态), 2 为 Tripod (三角步态)
        """
        rospy.loginfo("执行步态: {}".format("Ripple" if gait_type == 1 else "Tripod"))
        self.jethexa.traveling(
                  gait=gait_type, 
                  stride=40.0, 
                  height=15.0, 
                  direction=0, 
                  rotation=0.0,
                  time=1, 
                  steps=0, 
                  interrupt=True,
                  relative_height=False)

    def stop(self):
        rospy.loginfo("停止运动")
        self.jethexa.traveling(gait=0)
    

if __name__ == "__main__":
    node = MovingNode()
    rospy.sleep(2) # 等待订阅生效
    
    # 记录初始值
    rospy.loginfo("开始监测实验数据...")
    
    # 执行 10 秒的前进并持续返回数据
    node.forward(gait_type=1) # 默认采用 Ripple 步态
    
    start_time = rospy.get_time()
    while not rospy.is_shutdown() and (rospy.get_time() - start_time < 10):
        node.get_realtime_stats() # 实时打印速度和角度
        rospy.sleep(0.5) # 每0.5秒采样一次
        
    node.stop()
    rospy.on_shutdown(node.stop)
