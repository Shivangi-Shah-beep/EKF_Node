#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math
from opencv_apps.msg import Point2DArrayStamped
import tf2_ros
from sensor_msgs.msg import CameraInfo
from measurement_model import MeasurementModel

class EKFLocalizationNode:
    def __init__(self):
        rospy.init_node('ekf_localization_node', anonymous=True)

        self.alpha = [0.01, 0.01, 0.01, 0.01]
  
        self.mu = np.zeros((3, 1))  # [x, y, theta]
        self.Sigma = np.eye(3) * 0.1  # Initial 3x3 covariance matrix

        self.last_time = None  # For calculating time step

        # Landmark positions
        self.landmarks = ['red', 'green', 'yellow', 'magenta', 'cyan']
        self.landmark_positions = {
            'red': (8.5, -5.0),
            'green': (8.5, 5.0),
            'yellow': (-11.5, 5.0),
            'magenta': (-11.5, -5.0),
            'cyan': (0.0, 0.0)
        }

        # Initialize transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera and transform parameters
        self.camera_params = None
        self.transform_params = None
        self.measurement_models = {}
        self.Z_meas = {color: None for color in self.landmarks}

        # Initialize publishers
        self.pose_pub = rospy.Publisher('/ekf_pose', PoseWithCovarianceStamped, queue_size=10)

        # Subscribers
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        rospy.Subscriber('/front/left/camera_info', CameraInfo, self.camera_info_callback)

        for color in self.landmarks:
            topic = f'/goodfeature_{color}/corners'
            rospy.Subscriber(topic, Point2DArrayStamped, self.landmark_callback, callback_args=color)

        # Wait for static information
        self.wait_for_static_information()

        # Initialize measurement models for each landmark
        self.initialize_measurement_models()

        rospy.loginfo("EKF Localization Node Initialized")

    def wait_for_static_information(self):
        rospy.loginfo("Waiting for camera info...")
        camera_info_msg = rospy.wait_for_message('/front/left/camera_info', CameraInfo)
        self.camera_info_callback(camera_info_msg)

        rospy.loginfo("Waiting for static transform...")
        self.get_static_transform()

    def initialize_measurement_models(self):
        if self.camera_params is None or self.transform_params is None:
            rospy.logerr("Camera or transform parameters missing. Shutting down.")
            rospy.signal_shutdown("Initialization failed")
            return

        for color in self.landmarks:
            landmark_params = {
                'x_l': self.landmark_positions[color][0],
                'y_l': self.landmark_positions[color][1],
                'r_l': 0.1,  # Example radius
                'h_l': 1.0   # Example height
            }
            self.measurement_models[color] = MeasurementModel(
                landmark_params, self.camera_params, self.transform_params
            )
        rospy.loginfo("Measurement models initialized for all landmarks.")

    def camera_info_callback(self, msg):
        self.camera_params = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5]
        }
        rospy.loginfo("Camera parameters received")

    def get_static_transform(self):
        try:
            self.tf_buffer.can_transform('base_link', 'front_camera', rospy.Time(0), rospy.Duration(10.0))
            transform = self.tf_buffer.lookup_transform('base_link', 'front_camera', rospy.Time(0))
            self.transform_params = {
                't_cx': transform.transform.translation.x,
                't_cy': transform.transform.translation.y,
                't_cz': transform.transform.translation.z
            }
            rospy.loginfo("Static transform obtained")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get static transform: {e}")
            rospy.signal_shutdown("Transform not available")

    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        current_time = rospy.Time.now().to_sec()

        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        self.motion_update(v, w, dt)
        self.measurement_update()

    def motion_update(self, v, w, dt):
        theta = self.mu[2, 0]

        if w != 0:
            self.mu[0, 0] += (-v / w) * math.sin(theta) + (v / w) * math.sin(theta + w * dt)
            self.mu[1, 0] += (v / w) * math.cos(theta) - (v / w) * math.cos(theta + w * dt)
            self.mu[2, 0] += w * dt
        else:
            self.mu[0, 0] += v * math.cos(theta) * dt
            self.mu[1, 0] += v * math.sin(theta) * dt

        self.mu[2, 0] = math.atan2(math.sin(self.mu[2, 0]), math.cos(self.mu[2, 0]))

        G_t = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1, v * math.cos(theta) * dt],
            [0, 0, 1]
        ])

        Q_t = np.array([
            [self.alpha[0] * v**2 + self.alpha[1] * w**2, 0, 0],
            [0, self.alpha[2] * v**2 + self.alpha[3] * w**2, 0],
            [0, 0, self.alpha[3] * w**2]
        ])

        self.Sigma = G_t @ self.Sigma @ G_t.T + Q_t
        self.publish_pose()

    def measurement_update(self):
        for color, z_meas in self.Z_meas.items():
            if z_meas is None:
                rospy.logwarn(f"No measurement received for {color}, using Z_pred as Z_meas.")
                model = self.measurement_models[color]
                z_pred, _, _, _ = model.measurement(self.mu.flatten(), observed_features=[], variances=[])
                z_meas = np.array(z_pred)
                variances = [10.0] * len(z_meas)  
            else:
                variances = [10.0] * len(z_meas.flatten())

            model = self.measurement_models[color]
            z_pred, z_actual, _, z_diff = model.measurement(
                self.mu.flatten(), observed_features=z_meas.flatten(), variances=variances
            )
            H = model.jacobian(self.mu.flatten())

            
            R = np.diag(variances)
            S = H @ self.Sigma @ H.T + R
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            
            self.mu += K @ z_diff.reshape(-1, 1)
            self.mu[2, 0] = math.atan2(math.sin(self.mu[2, 0]), math.cos(self.mu[2, 0]))  # Normalize theta

            
            self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

            rospy.loginfo(f"Updated state for {color}: {self.mu.flatten()}")

                

    def landmark_callback(self, msg, color):
        if len(msg.points) < 4:
            self.Z_meas[color] = None
            return

        self.Z_meas[color] = np.array([[pt.x, pt.y] for pt in msg.points])

    def publish_pose(self):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'map'

        # Position and Orientation
        pose_msg.pose.pose.position.x = self.mu[0, 0]
        pose_msg.pose.pose.position.y = self.mu[1, 0]
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation.z = math.sin(self.mu[2, 0] / 2)
        pose_msg.pose.pose.orientation.w = math.cos(self.mu[2, 0] / 2)

        # Expanded Covariance
        covariance_6x6 = np.zeros((6, 6))
        covariance_6x6[:2, :2] = self.Sigma[:2, :2]
        covariance_6x6[5, 5] = self.Sigma[2, 2]
        pose_msg.pose.covariance = covariance_6x6.flatten().tolist()
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Pose published successfully.")
        


def main():
    try:
        ekf_node = EKFLocalizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
