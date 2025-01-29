#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Twist.h>

class BayesFilter {
public:
    BayesFilter() {
        // Initialize the ROS node
        ros::NodeHandle nh;

        // Publishers
        door_open_pub_ = nh.advertise<std_msgs::Empty>("/door_open", 10);
        cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        door_torque_pub_ = nh.advertise<std_msgs::Float64>("/door_torque", 10);

        // Subscriber
        feature_mean_sub_ = nh.subscribe("/feature_mean", 10, &BayesFilter::callback, this);

        // Initialize beliefs and probabilities
        initBelief();
    }

    void initBelief() {
        bel_open_ = 0.5;
        bel_close_ = 0.5;

        // Transition Model
        open_command_open_ = 1.0;
        close_command_open_ = 0.0;
        open_command_close_ = 0.16;
        close_command_close_ = 0.84;

        // Observation Model
        open_given_open_ = 0.875;
        open_given_closed_ = 0.275;
        closed_given_closed_ = 0.725;
        closed_given_open_ = 0.125;

        feature_mean_ = -1;
        threshold_ = 446.0;
        measurement_ = "unknown";
    }

    void callback(const std_msgs::Float64::ConstPtr& msg) {
        ROS_INFO("Callback triggered");
        feature_mean_ = msg->data;
        ROS_INFO("Received sensor data: %f", feature_mean_);
        bayesFilter();
    }

    void updateMeasurement() {
        if (feature_mean_ < 0) {
            ROS_WARN("Feature mean is invalid. Skipping measurement update.");
            return;
        }

        measurement_ = (feature_mean_ < threshold_) ? "open" : "closed";
        ROS_INFO("Measurement updated: %s", measurement_.c_str());
    }

    void openDoor() {
        ROS_INFO("Publishing empty message to /door_open (open door)");
        std_msgs::Empty msg;
        door_open_pub_.publish(msg);
        ros::Duration(0.5).sleep();
    }

    void bayesFilter() {
        // Update the measurement based on feature_mean
        openDoor();
        updateMeasurement();

        // Predict step
        double bel_open_pred = (bel_open_ * open_command_open_) + (bel_close_ * open_command_close_);
        double bel_close_pred = (bel_close_ * close_command_close_) + (bel_open_ * close_command_open_);
        ROS_INFO("Action update: bel_open=%.2f, bel_close=%.2f", bel_open_pred, bel_close_pred);

        // Measurement update
        double numerator, denominator;
        if (measurement_ == "open") {
            numerator = open_given_open_ * bel_open_pred;
            denominator = (open_given_open_ * bel_open_pred) + (open_given_closed_ * bel_close_pred);
        } else { // measurement_ == "closed"
            numerator = closed_given_open_ * bel_open_pred;
            denominator = (closed_given_open_ * bel_open_pred) + (closed_given_closed_ * bel_close_pred);
        }

        if (denominator == 0) {
            ROS_WARN("Denominator is zero. Skipping belief update.");
            return;
        }

        // Update beliefs
        bel_open_ = numerator / denominator;
        bel_close_ = 1.0 - bel_open_;
        ROS_INFO("Measurement update: bel_open=%.2f, bel_close=%.2f", bel_open_, bel_close_);

        // Check if the belief threshold is met
        if (bel_open_ >= 0.99) {
            ROS_INFO("Threshold achieved");
            move(2.0, 5.0); // Move through the door for 5 seconds
            ros::shutdown(); // Shut down the node
        }
    }

    void move(double speed, double duration) {
        ROS_INFO("Moving robot at speed %.2f for %.2f seconds", speed, duration);
        geometry_msgs::Twist twist;
        ros::Rate rate(10); // 10 Hz loop rate
        ros::Time start_time = ros::Time::now();

        while ((ros::Time::now() - start_time).toSec() < duration) {
            twist.linear.x = speed;
            cmd_pub_.publish(twist);
            rate.sleep();
        }

        // Stop the robot
        twist.linear.x = 0.0;
        cmd_pub_.publish(twist);
        ROS_INFO("Robot movement stopped");
    }

private:
    ros::Publisher door_open_pub_;
    ros::Publisher cmd_pub_;
    ros::Publisher door_torque_pub_;
    ros::Subscriber feature_mean_sub_;

    // Beliefs and probabilities
    double bel_open_;
    double bel_close_;
    double open_command_open_;
    double close_command_open_;
    double open_command_close_;
    double close_command_close_;
    double open_given_open_;
    double open_given_closed_;
    double closed_given_closed_;
    double closed_given_open_;

    double feature_mean_;
    double threshold_;
    std::string measurement_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "bayes_filter_3");
    ROS_INFO("Starting Bayes Filter Node...");
    BayesFilter node;
    ros::spin();
    ROS_INFO("Bayes Filter Node shutting down.");
    return 0;
}

