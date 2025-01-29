#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty, Float64
from geometry_msgs.msg import Twist

class BayesFilter:
    def __init__(self):
        rospy.init_node('bayes_filter_3', anonymous=True)

        # Publishers
        self.door_open_pub = rospy.Publisher('/door_open', Empty, queue_size=10)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.door_torque_pub = rospy.Publisher('/door_torque', Float64, queue_size=10)

        # Subscribers
        self.sub = rospy.Subscriber('/feature_mean', Float64, self.callback)

        # Initialize beliefs and probabilities
        self.init_belief()
        self.feature_mean = None
        self.threshold = 452
        self.measurement = None

    def init_belief(self):
        self.bel_open = 0.5
        self.bel_close = 0.5

        # Transition Model
        self.open_command_open = 1.0
        self.close_command_open = 0.0
        self.open_command_close = 0.16
        self.close_command_close = 0.84

        # Observation Model
        self.open_given_open = 0.875
        self.open_given_closed = 0.275
        self.closed_given_closed = 0.725
        self.closed_given_open = 0.125

    def callback(self, msg):
        rospy.loginfo("Callback triggered")
        self.feature_mean = msg.data
        rospy.loginfo(f"Received sensor data: {self.feature_mean}")
        self.bayes_filter()

    def update_measurement(self):
        if self.feature_mean is None:
            rospy.logwarn("Feature mean is None. Skipping measurement update.")
            return
        self.measurement = 'open' if self.feature_mean < self.threshold else 'closed'
        rospy.loginfo(f"Measurement updated: {self.measurement}")

    def open_door(self):
        rospy.loginfo("Publishing empty message to /door_open (open door)")
        self.door_open_pub.publish(Empty())
        rospy.sleep(0.5)

    def bayes_filter(self):
        # Update the measurement based on feature_mean
        self.open_door()
        self.update_measurement()

        # Predict step
        bel_open_pred = (self.bel_open * self.open_command_open) + (self.bel_close * self.open_command_close)
        bel_close_pred = (self.bel_close * self.close_command_close) + (self.bel_open * self.close_command_open)
        rospy.loginfo(f"Action update: bel_open={bel_open_pred:.2f}, bel_close={bel_close_pred:.2f}")

        # Measurement update
        if self.measurement == 'open':
            numerator = self.open_given_open * bel_open_pred
            denominator = (self.open_given_open * bel_open_pred) + (self.open_given_closed * bel_close_pred)
        else:  # self.measurement == 'closed'
            numerator = self.closed_given_open * bel_open_pred
            denominator = (self.closed_given_open * bel_open_pred) + (self.closed_given_closed * bel_close_pred)

        if denominator == 0:
            rospy.logwarn("Denominator is zero. Skipping belief update.")
            return

        # Update beliefs
        self.bel_open = numerator / denominator
        self.bel_close = 1 - self.bel_open
        rospy.loginfo(f"Measurement update: bel_open={self.bel_open:.2f}, bel_close={self.bel_close:.2f}")

        # Check if the belief threshold is met
        if self.bel_open >= 0.99:
            rospy.loginfo("Threshold achieved")
            self.move(2, duration=5) 
            rospy.signal_shutdown("Node terminated.")


    def move(self, speed, duration):
        rate = rospy.Rate(10)  # 10 Hz loop rate
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            twist = Twist()
            twist.linear.x = speed
            self.cmd_pub.publish(twist)
            rate.sleep()

def main():
    rospy.loginfo("Starting Bayes Filter Node...")
    node=BayesFilter()
    node.bayes_filter()
    rospy.spin()
    rospy.loginfo("Bayes Filter Node shutting down.")

if __name__ == '__main__':
    main()
