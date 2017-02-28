/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "Utility.h"

//ROS
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_msgs/TFMessage.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <kdl/frames_io.hpp>

//PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

void HSVtoRGB(double &r, double &g, double &b, double h, double s, double v) {
    int i;
    double f, p, q, t;

    if (s == 0) {
        // achromatic (grey)
        r = g = b = v;
        return;
    }

    h /= 60; // sector 0 to 5
    i = floor(h);
    f = h - i; // factorial part of h
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch (i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default: // case 5:
            r = v;
            g = p;
            b = q;
            break;
    }

}



int main(int argc, char** argv) {
    ros::init(argc, argv, "my_tf_broadcaster");
    ros::NodeHandle node;

    //ros::Publisher cloud_pub = node.advertise<sensor_msgs::PointCloud2>("/output_pcd", 1);
    pcl::PointCloud<PointType>::Ptr cloud_voxel_rviz(new pcl::PointCloud<PointType>);
    sensor_msgs::PointCloud2 cloud_msg;
    
    pcl::io::loadPCDFile<PointType> ("output.pcd", *cloud_voxel_rviz);   
    pcl::toROSMsg(*cloud_voxel_rviz, cloud_msg);
    cloud_msg.header.frame_id = "/cloud_frame";

    int size = (int) cloud_voxel_rviz->size();

    ros::Publisher marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    visualization_msgs::Marker marker;

    marker.header.frame_id = "cloud";
    marker.header.stamp = ros::Time::now();
    marker.ns = "basic_shapes";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0; 
    marker.pose.position.y = 0; 
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.1;
    marker.color.b = 0.0;
    marker.points.resize(size);
    marker.colors.resize(size);


    double H, R, G, B;
    for (int i = 0; i < size; i++) {
        marker.points[i].x = cloud_voxel_rviz->points[i].x;
        marker.points[i].y = cloud_voxel_rviz->points[i].y;
        marker.points[i].z = cloud_voxel_rviz->points[i].z;

        H = 0 + (2 - marker.points[i].z) / 2 * 180; //choose the range of H from 0 to 180,z is from 0-2
        HSVtoRGB(R, G, B, H, 1.0, 1.0);
        
        marker.colors[i].r = R;
        marker.colors[i].g = G;
        marker.colors[i].b = B;
        marker.colors[i].a = 1.0;

    }


    ros::Rate r(1);
    while (ros::ok()) {
        
        marker_pub.publish(marker);     
        //cloud_pub.publish(cloud_msg);

        ros::spinOnce();
        r.sleep();
    }
};