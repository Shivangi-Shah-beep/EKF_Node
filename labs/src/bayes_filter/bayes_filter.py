#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64, Empty

class Door():
    def __init__(self):
        # Create a publisher for the door torque
        self.door_open = rospy.Publisher('/door_open', Empty, queue_size=10)
        self.door_torque_sub = rospy.Subscriber('/hinged_glass_door/torque', Float64, self.callback)
        self.door_torque_pub = rospy.Publisher('/hinged_glass_door/torque', Float64, queue_size=10)
        rospy.sleep(1)
        self.X = 'closed'
        self.torque = 0
        self.new_torque_received = False  # Flag to check if new torque is received

    def callback(self, msg):
        """ ROS subscriber callback to update torque value """
        self.torque = msg.data
        self.new_torque_received = True  # Set flag when new torque is received
        rospy.loginfo(f"Received Torque: {self.torque}")

    def check_open_or_closed(self):
        """ Check whether the door is open or closed based on torque """
        if self.torque > 1.5:
            self.X = 'open'
        else:
            self.X = 'closed'

    def close(self):
        """ Close the door by applying negative torque """
        torque = 10
        self.door_torque_pub.publish(Float64(-torque))
        rospy.loginfo("The door has been closed again.")

    def publish_door(self):
        """ Publish an empty message to attempt opening the door """
        rospy.loginfo("Publishing empty message to /door_open")
        self.door_open.publish(Empty())

def main():
    rospy.init_node('bayes_filter', anonymous=True)

    num_trials = 50
    door_open_count = 0  # Counter for how many times the door opens
    door_closed_count = 0  # Counter for how many times the door stays closed

    door = Door()

    for _ in range(num_trials):
        door.publish_door()  # Send command to open the door
        rospy.sleep(0.3)  # Slightly increased wait time

        # Wait until a new torque value is received
        door.new_torque_received = False
        wait_time = 0
        while not door.new_torque_received and wait_time < 1.0:  # Wait max 1 second
            rospy.sleep(0.1)
            wait_time += 0.1

        door.check_open_or_closed()  # Check if the door opened

        rospy.loginfo(f"The torque is {door.torque}")

        if door.X == 'open':
            door_open_count += 1
            rospy.loginfo("Door has opened")
            door.close()  # Close the door
            rospy.sleep(0.2)
        else:
            door_closed_count += 1
            rospy.loginfo("Door is closed. Trying again")

    # Final report
    rospy.loginfo(f"The door opened {door_open_count} times out of {num_trials} trials.")
    rospy.loginfo(f"The door remained closed {door_closed_count} times out of {num_trials} trials.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
