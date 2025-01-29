#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64

class Door():
    def __init__(self):
        # Create a publisher for the door torque
        self.door_torque_pub = rospy.Publisher('/hinged_glass_door/torque', Float64, queue_size=10)
        # Ensure publisher is ready
        rospy.sleep(1)
        self.X = 'closed'  # Initialize door as closed

    def open(self, torque):
        # Publish the torque to open the door
        rospy.loginfo(f'Publishing torque {torque} to open the door...')
        self.door_torque_pub.publish(Float64(torque))
        self.X = 'open'
        rospy.loginfo('Door is now open.')

    def close(self, torque):
        # Publish negative torque to close the door
        rospy.loginfo(f'Publishing torque {-torque} to close the door...')
        self.door_torque_pub.publish(Float64(-torque))
        self.X = 'closed'
        rospy.loginfo('Door is now closed.')

class Z():
    def __init__(self):
        self.feature_mean = None  
        self.sub = rospy.Subscriber('/feature_mean', Float64, self.callback)
        self.signal = 'none'  

    def callback(self, feature_mean):
        self.feature_mean = feature_mean.data  
    
    def check_measurement(self):
        self.threshold = 451.42
        if self.feature_mean is None:
            rospy.loginfo("Waiting for feature_mean data...")
            return None  # No data received yet
        if self.feature_mean > self.threshold: 
            self.signal = 'open'
        else: 
            self.signal = 'closed'

def main():
    rospy.init_node('prob_bayes_filter', anonymous=True)
    
    # Initialize counters for conditional probabilities
    open_open = 0.0  # P(z=open | x=open)
    open_closed = 0.0  # P(z=open | x=closed)
    closed_closed = 0.0  # P(z=closed | x=closed)
    closed_open = 0.0  # P(z=closed | x=open)

    door = Door()
    z = Z()

    num_trials = 40  # Number of trials
    for trials in range(num_trials):
        torque_value = 10.0  

        # Open the door
        rospy.loginfo("Opening the door...")
        door.open(torque_value)
        rospy.sleep(0.5)  

        # Call the check_measurement function to update the signal
        while z.feature_mean is None:
            rospy.sleep(0.1)  

        z.check_measurement()

        # Check if the door is open
        if door.X == 'open':
            # Check what the measurement says
            if z.signal == 'open':
                # Z=open|x=open
                open_open += 1
                rospy.loginfo("Door is open and measured open")
            elif z.signal == 'closed':
                # Z=closed|X=open
                closed_open += 1
                rospy.loginfo("Door is open and measured closed")

        rospy.sleep(5)

        # Close the door
        rospy.loginfo("Closing the door...")
        door.close(torque_value)
        rospy.sleep(0.5)  # Wait a bit to allow the callback to update feature_mean

        # Call the check_measurement function to update the signal
        while z.feature_mean is None:
            rospy.sleep(0.1)  

        z.check_measurement()

        # Check if the door is closed
        if door.X == 'closed':
            # Check what the measurement says
            if z.signal == 'open':
                # Z=open|x=closed
                open_closed += 1
                rospy.loginfo("Door is closed and measured open")
            elif z.signal == 'closed':
                # Z=closed|X=closed
                closed_closed += 1
                rospy.loginfo("Door is closed and measured closed")

        rospy.sleep(5)
            
    
    open_open /= num_trials
    open_closed /= num_trials
    closed_closed /= num_trials
    closed_open /= num_trials

    # Log the probabilities
    rospy.loginfo(f"P(z=open | x=open): {open_open}")
    rospy.loginfo(f"P(z=open | x=closed): {open_closed}")
    rospy.loginfo(f"P(z=closed | x=closed): {closed_closed}")
    rospy.loginfo(f"P(z=closed | x=open): {closed_open}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
