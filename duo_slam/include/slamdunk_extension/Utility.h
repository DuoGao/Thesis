/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Utility.h
 * Author: Duo Gao
 *
 * Created on 13 luglio 2016, 15.35
 */

#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iomanip>
#include <iostream>

#include <fstream>
#include <map>
#include <vector>
#include <kdl/frames_io.hpp>
#include <tf/tf.h>
#include <eigen3/Eigen/Core>
#include "geometry_msgs/Pose.h"

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/surface/convex_hull.h>
#include <nav_msgs/Odometry.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

//tf
#include <tf2_msgs/TFMessage.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#define Radius 2
typedef pcl::PointXYZRGB PointType;
#define Voxel_grid_rows 200
#define Voxel_grid_cols 200
#define Voxel_grid_higs 40

#define unit_rows  0.05
#define unit_cols 0.05
#define unit_height 0.05

#define static_allocation 0
#define dynamic_allocation 1 

#define mode_obs 1 
#define mode_plane 0

#define integrate_mode 1
#define deintegrate_mode -1

//below only for the 3D visualizer
#define stepSize_rows 4
#define stepSize_cols 4 
#define thre_counter 3

//for linked list
#define list_num 5

class Utility {
public:
    Utility();
    Utility(const Utility& orig);
    virtual ~Utility();

    static void kdl_to_eigen_4x4_d(KDL::Frame& frame, Eigen::Matrix4d& mat);
    static void create_kdl_frame(float x, float y, float z, float roll, float pitch, float yaw, KDL::Frame& out_frame);
    static void create_eigen_4x4_d(float x, float y, float z, float roll, float pitch, float yaw, Eigen::Matrix4d& mat);
    static void eigen_4x4_to_geometrypose_d(Eigen::Matrix4d& mat,geometry_msgs::Pose& pose);
    static void eigen_4x4_d_to_tf(Eigen::Matrix4d& t,  tf::Transform& tf, bool reverse);
    static void draw_reference_frame(pcl::visualization::PCLVisualizer &viewer,  Eigen::Matrix4d& rf, float size, std::string name);
    static void convert_point_3D(PointType& pt, Eigen::Vector3f& p, bool reverse);
    static void draw_3D_vector(pcl::visualization::PCLVisualizer& viewer, Eigen::Vector3f start, Eigen::Vector3f end, float r, float g, float b, std::string name);
    
private:

};

#endif /* UTILITY_H */

