#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iomanip>

//ROS
#include <ros/ros.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud2.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <kdl/frames_io.hpp>
#include "geometry_msgs/Pose.h"
#include <std_msgs/UInt32.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>


//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//LAR_TOOLS
#include "lar_tools.h"
#include "lar_vision/reconstruction/tsdf/tsdf_volume_octree.h"


#include <boost/thread/thread.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>


//defines
typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
using namespace std;

ros::NodeHandle* nh;

//Topics
ros::Publisher pointcloud_pub;

/** MAIN NODE **/
int main(int argc, char** argv) {

    // Initialize ROS
    ros::init(argc, argv, "slam_test");
    ROS_INFO("slam_test node started...");
    nh = new ros::NodeHandle("~");

    int hz;
    nh->param<int>("hz", hz, 30);

    // Spin & Time
    ros::Rate r(hz);

    ros::Publisher pointcloud_pub = nh->advertise<sensor_msgs::PointCloud2>("/trial", 1);


    sensor_msgs::PointCloud2 cloud_msg;
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    int w = 100;
    cloud->width=w*w;
    cloud->height=1;

    for (int i = 0; i < w*w; i++) {
        int x = i % w;
        int y = i / w;
        PointType p;
        p.x = x*0.1;
        p.y = y*0.1;
        p.z = 1;
        p.r = 255;
        p.g = 255;
        p.b = rand() * 255;
        cloud->points.push_back(p);
    }
    
    pcl::toROSMsg(*cloud,cloud_msg);
    cloud_msg.header.frame_id = "base";
    // Spin
    while (nh->ok()) {
        
        pointcloud_pub.publish(cloud_msg);
        

        ros::spinOnce();
        r.sleep();
    }

}
