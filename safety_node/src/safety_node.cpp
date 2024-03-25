#include "rclcpp/rclcpp.hpp"
/// CHECK: include needed ROS msg type headers and libraries
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <math.h>
#include <algorithm>
#include <chrono>


using std::placeholders::_1;
using namespace std::chrono_literals;


class Safety : public rclcpp::Node {
// The class that handles emergency braking

public:
    Safety() : Node("safety_node")
    {
        /*
        You should also subscribe to the /scan topic to get the
        sensor_msgs/LaserScan messages and the /ego_racecar/odom topic to get
        the nav_msgs/Odometry messages

        The subscribers should use the provided odom_callback and 
        scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        */

        /// TODO: create ROS subscribers and publishers
        laser_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan",100,std::bind(&Safety::scan_callback,this,_1));
        stop_publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive",100);
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom",100,std::bind(&Safety::drive_callback,this,_1));
    }

private:
    double speed = 0.0;
    double range_rate;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr stop_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;

    /// TODO: create ROS subscribers and publishers

    void drive_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        /// TODO: update current speed
        speed = msg->twist.twist.linear.x;
        //RCLCPP_INFO(this->get_logger(),"Current speed: %f",speed);
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        /// TODO: calculate TTC

        /// TODO: publish drive/brake message
        std::vector<float> ranges = scan_msg->ranges;
        int ranges_len = ranges.size();
        std::vector<float> ittc(ranges_len,-1);
        auto inf = std::numeric_limits<float>::infinity();
        double angle_increment = (180/M_PI)*scan_msg->angle_increment;
        double angle_ini = (180/M_PI)*scan_msg->angle_min;
        //double angle_max = (180/M_PI)*scan_msg->angle_max;
        for (int i=0;i<ranges_len;i++){
            if(ranges[i] == inf || isnan(ranges[i])) continue;
            angle_ini += angle_increment;
            if (abs(angle_ini+angle_increment) <= 30) {
                double range_rate = speed*cos(angle_ini*M_PI/180.0);
                ittc[i] = ranges[i] / std::max(0.0,range_rate);
                
                //RCLCPP_INFO(this->get_logger(),"Current range: %f, acos_ngle_ini: %f, range_rate: %f, speed: %f, angle_ iitc: %f",ranges[i],cos(angle_ini*M_PI/180.0),range_rate,speed,ittc[i]);
                if (ittc[i] <0.8) {
                    RCLCPP_INFO(this->get_logger(),"Applying brakes! ittc: %f , range: %f ",ittc[i],ranges[i]);
                    //timer_ = this->create_wall_timer(
                    //    10ms, std::bind(&Safety::brake_timer_callback, this));
                    brake_callback();
                    break;
                }
            }
        }

    }

    void brake_callback(){

        auto message = ackermann_msgs::msg::AckermannDriveStamped();
        message.drive.speed = 0;
        stop_publisher_->publish(message);

    }



};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}
