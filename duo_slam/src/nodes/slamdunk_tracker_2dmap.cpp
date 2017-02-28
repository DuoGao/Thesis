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
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <kdl/frames_io.hpp>
#include "geometry_msgs/Pose.h"
#include <std_msgs/UInt32.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf2_msgs/TFMessage.h>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>


//Slamdunk
#include <slamdunk/slam_dunk.h>
#include <slamdunk_extension/SlamDunkScene.h>

#include <boost/thread/thread.hpp>
#include <geometry_msgs/PoseStamped.h>

//Two_D_Map///////////////////////////////////
#include <slamdunk_extension/Two_D_Map.h>
#include <slamdunk_extension/Map_3D.h>
#include <slamdunk_extension/tf_listener.h>
/////////////////////////////////////////////
using namespace std;

ros::NodeHandle* nh;

//Topics
ros::Publisher pointcloud_pub;
ros::Publisher camera_pose_pub;
ros::Publisher camera_odometry_pub;
double current_time, start_time;
double tracker_start_time;

//Clouds&Viewer
pcl::visualization::PCLVisualizer* viewer;

//SlamDunk
boost::shared_ptr<slamdunk::SlamDunk> slam_dunk;
slamdunk::SlamDunkScene slam_dunk_scene;

//Transforms
Eigen::Matrix4d T_0_S;
Eigen::Matrix4d T_S_CAM;
Eigen::Matrix4d T_0_CAM;
Eigen::Matrix4d T_CAMERA_CAMERACORR;
//Messages
geometry_msgs::PoseStamped camera_pose;
nav_msgs::Odometry camera_odometry;

//Params
float fx, fy, cx, cy, cols, rows, depth_scale_factor;
bool build_environment = false;
bool viewer_to_update = false;

//Frame & Buffer

struct BufferTime {
    int secs;
    int nsecs;

    BufferTime() {
        secs = 0;
        nsecs = 0;
    }

    BufferTime(int secs, int nsecs) {
        this->secs = secs;
        this->nsecs = nsecs;
    }
};

//Buffer attributes
Eigen::Matrix3f inverse_kcam;
cv::Mat current_frame_rgb;
cv::Mat current_frame_depth;
bool rgb_frame_ready = false;
bool depth_frame_ready = false;
slamdunk::RGBDFrame current_frame_rgbd;
const int frame_buffer_size = 2;
int frame_buffer_index = 0;
int frame_ready_to_consume = -1;
bool first_frame_ready = false;
slamdunk::RGBDFrame frame_buffer[frame_buffer_size];
BufferTime frame_buffer_time[frame_buffer_size];

//Locks
boost::mutex buffer_lock;
boost::mutex view_mutex;
boost::condition_variable buffer_lock_variable;

//My parm     £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
Two_D_Map two_d_map;
Map_3D map_3d;
tf_listener* TF_lis;
//Utility utility;
pcl::PointCloud<PointType>::Ptr cloud_voxel_rviz(new pcl::PointCloud<PointType>);
bool First_callback_lock = true;
//£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££ 

/**
 * Slamdunk scene update by KEY
 */
void update_cloud_view(slamdunk::SlamDunkScene& slam_dunk_scene, std::string key) {

    if (slam_dunk_scene.existsCloud(key)) {
        boost::mutex::scoped_lock lock(view_mutex);
        viewer->removePointCloud(key);

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(*(slam_dunk_scene.clouds[key]), *cloud, slam_dunk_scene.poses[key].matrix());

        viewer->addPointCloud(cloud, key, 1);

        //My parm£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
        std::string plane_name = key + "_plane";
        std::string rest_name = key + "_rest";
        viewer->removePointCloud(plane_name);
        viewer->removePointCloud(rest_name);

        pcl::PointCloud<PointType>::Ptr cloud_P(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr cloud_R(new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(two_d_map.MAP_PLANE[key], *cloud_P, slam_dunk_scene.poses[key].matrix());
        pcl::transformPointCloud(two_d_map.MAP_REST[key], *cloud_R, slam_dunk_scene.poses[key].matrix());

        pcl::visualization::PointCloudColorHandlerCustom<PointType> red(cloud_R, 255, 0, 0);
        viewer->addPointCloud(cloud_R, red, rest_name, 2);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> white(cloud_P, 255, 255, 255);
        viewer->addPointCloud(cloud_P, white, plane_name, 2);
        //££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££     
    }
}

/**
 * RGB + DEPTH callback
 */
void callback(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth) {

    //RGB
    current_frame_rgbd.m_color_image = cv_bridge::toCvShare(rgb, "bgr8")->image;

    //DEPTH
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(depth);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    current_frame_depth = cv_ptr->image;
    current_frame_rgbd.m_depth_image = cv::Mat(current_frame_depth.rows, current_frame_depth.cols, CV_32FC1);
    for (int j = 0; j < current_frame_depth.cols; j++) {
        for (int i = 0; i < current_frame_depth.rows; i++) {
            current_frame_rgbd.m_depth_image.at<float>(i, j) = (float) (current_frame_depth.at<unsigned short>(i, j) / 1000.0f);
        }
    }

    //Store Frame in buffer
    boost::lock_guard<boost::mutex> lock(buffer_lock);
    current_frame_rgbd.m_timestamp = rgb->header.stamp.nsec * 10e-9;
    frame_buffer[frame_buffer_index] = current_frame_rgbd;
    frame_buffer_time[frame_buffer_index] = BufferTime(depth->header.stamp.sec, depth->header.stamp.nsec);
    buffer_lock.unlock();

    first_frame_ready = true;
}

/**
 * Create the FeatureTracke by string naming the Feature Extractor/Detector
 */
void create_feature_tracker(slamdunk::FeatureTrackerParams& ft_params, std::string feature_str) {
    if (feature_str == "SURF" || feature_str == "surf") {
        ft_params.feature_extractor = cv::Ptr<const cv::Feature2D>(new cv::SURF(500, 4, 2, false, true));
    } else if (feature_str == "SURFEXT" || feature_str == "surfext") {
        ft_params.feature_extractor = cv::Ptr<const cv::Feature2D>(new cv::SURF(500, 4, 2, true, false));
    } else if (feature_str == "SIFT" || feature_str == "sift") {
        ft_params.feature_extractor = cv::Ptr<const cv::Feature2D>(new cv::SIFT());
    } else if (feature_str == "ORB" || feature_str == "orb") {
        ft_params.feature_extractor = cv::Ptr<const cv::Feature2D>(new cv::ORB());
        ft_params.feature_matcher = slamdunk::FeatureMatcher::Ptr(new slamdunk::RatioMatcherHamming(false, 0.8, ft_params.cores));
    } else if (feature_str == "BRISK" || feature_str == "brisk") {
        ft_params.feature_extractor = cv::Ptr<const cv::Feature2D>(new cv::BRISK());
        ft_params.feature_matcher = slamdunk::FeatureMatcher::Ptr(new slamdunk::RatioMatcherHamming(false, 0.8, ft_params.cores));
    } else
        std::cout << ">> Features ``" << feature_str << "'' not supported." << std::endl;
}

/**
 * SlamDunk Loop 
 */
void slam_dunk_loop() {
    Eigen::Isometry3d current_pose;

    while (ros::ok()) {
        if (current_time > tracker_start_time && first_frame_ready) {

            boost::lock_guard<boost::mutex> lock(buffer_lock);

            //Consume from buffer
            slamdunk::RGBDFrame frame = slamdunk::cloneRGBDFrame(frame_buffer[frame_buffer_index]);
            BufferTime frame_time = frame_buffer_time[frame_buffer_index];
            frame_buffer_index++;
            frame_buffer_index = frame_buffer_index % frame_buffer_size;

            buffer_lock.unlock();

            //Check if Frame is void //TODO: this must be never true!
            if (frame.m_color_image.empty() || frame.m_depth_image.empty())continue;

            //Slamdunk thick!
            const int tracked = (*slam_dunk)(frame, current_pose);

            //camera pose
            T_S_CAM = current_pose.matrix();
            //Camera pose publisher
            T_0_CAM = T_0_S*T_S_CAM;

            //Update Environment
            if (build_environment)
                if (tracked == slamdunk::SlamDunk::KEYFRAME_DETECTED) {

                    std::string key = slam_dunk_scene.getKeyframeName(frame.m_timestamp);

                    pcl::PointCloud<PointType>::Ptr cloud = slam_dunk_scene.createCloudFromFrame(frame, fx, fy, cx, cy, 8);
                    slam_dunk_scene.addCloud(key, cloud);
                    slam_dunk_scene.addPose(key, current_pose);
                    slam_dunk_scene.addRGBDFrame(key, frame);
                    slam_dunk_scene.updatePoseHistory(key, current_pose, true);

                    //My parm£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££   
                    if (First_callback_lock) {
                        two_d_map.T_S_CAMFIRST = T_S_CAM; //the first position seen in the slamdunk
                        two_d_map.first_callback_init(slam_dunk_scene.clouds[key]);
                        two_d_map.init_Voxel_grid();

                        TF_lis->init_T_Asus_cami();
                        TF_lis->cami_VO_eigen(TF_lis->T_vicon_Asusfirst, TF_lis->T_odom_camfirst); //the first position seen in the vicon and odom
                        TF_lis->T_vicon_camfirst = TF_lis->T_vicon_Asusfirst * TF_lis->T_Asus_cami;

                        map_3d.T_S_CAMFIRST = T_S_CAM;
                        map_3d.T_CT_CAMFIRST = two_d_map.T_CAMFIRST_CT.inverse(); //for odom 3D      

                        First_callback_lock = false;
                    }
                    //££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££


                    update_cloud_view(slam_dunk_scene, key);
                    viewer_to_update = true;
                }

            Utility::eigen_4x4_to_geometrypose_d(T_0_CAM, camera_pose.pose);
            camera_pose_pub.publish(camera_pose);

            //Publish Tracking information
            if (tracked == slamdunk::SlamDunk::FRAME_TRACKED) {
                camera_odometry.child_frame_id = "TRACKED";
            } else if (tracked == slamdunk::SlamDunk::KEYFRAME_DETECTED) {
                camera_odometry.child_frame_id = "KEYFRAME";
            } else if (tracked == slamdunk::SlamDunk::TRACKING_FAILED) {
                camera_odometry.child_frame_id = "FAILED";
            }
            camera_odometry.header.stamp.sec = frame_time.secs; // ros::Time::now().sec;
            camera_odometry.header.stamp.nsec = frame_time.nsecs; // ros::Time::now().nsec;
            Utility::eigen_4x4_to_geometrypose_d(T_0_CAM, camera_odometry.pose.pose);
            camera_odometry_pub.publish(camera_odometry);

        }
    }
}

/**
 * Checks for Optimized frames
 */
void check_optimized_frames() {

    std::string key;

    while (ros::ok()) {

        if (First_callback_lock) continue;

        std::vector<slamdunk::ConsumingEntry> entries;
        slam_dunk_scene.getAvailableEntries(entries);

        for (int i = 0; i < entries.size(); i++) {

            key = entries[i].key;

            if (!entries[i].replacement) {
                ROS_INFO("Integrate new entry!");
//My parm£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
                //Creation of clouds
                two_d_map.pass_through_fillter(*(slam_dunk_scene.clouds[key]), 0.2, -0.2, true);             
                two_d_map.ground_finder(slam_dunk_scene.clouds[key], key, T_S_CAM, 30.0, 0.3, 0.05);
                //two_d_map.RadiusOutlierRemoval(two_d_map.MAP_REST[key], 0.5, 200);              
                //two_d_map.project_to_1stplane(two_d_map.MAP_REST[key]);
                //two_d_map.compute_Voxel_grid(two_d_map.MAP_REST[key], mode_obs);
                //two_d_map.compute_Voxel_grid(two_d_map.MAP_PLANE[key], mode_plane);

                //Manage trajectories tfs
                TF_lis->vicon_cami_in_1stref(two_d_map.T_CAMFIRST_CT);
                TF_lis->odom_cami_in_1stref(two_d_map.T_CAMFIRST_CT);

                //Integration in 3D Voxel Grid
                map_3d.integrate_clouds(entries[i].new_pose, two_d_map.MAP_REST[key], two_d_map.MAP_PLANE[key], integrate_mode); //False = integrate clouds

                //update the trajectory 
                map_3d.update_trajectory(entries[i].new_pose, key);
//££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
                update_cloud_view(slam_dunk_scene, entries[i].key);

            } else {

                ROS_INFO("Replace old entry!");
//My parm£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
             
                //Deintegration in 3D Voxel Grid
                map_3d.integrate_clouds(entries[i].old_pose, two_d_map.MAP_REST[key], two_d_map.MAP_PLANE[key], deintegrate_mode); //TRue = DEintegrate clouds
                map_3d.integrate_clouds(entries[i].new_pose, two_d_map.MAP_REST[key], two_d_map.MAP_PLANE[key], integrate_mode); //False = integrate clouds
                
                //update the trajectory 
                map_3d.update_trajectory(entries[i].new_pose, key);

//££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
                slam_dunk_scene.addPose(entries[i].key, entries[i].new_pose);
                update_cloud_view(slam_dunk_scene, entries[i].key);
            }
        }
    }
}

/** MAIN NODE **/
int main(int argc, char** argv) {

    // Initialize ROS
    ros::init(argc, argv, "slam_test");
    ROS_INFO("slam_test node started...");
    nh = new ros::NodeHandle("~");

    int hz;
    nh->param<int>("hz", hz, 30);

    int cores;
    nh->param<int>("cores", cores, 4);

    bool viz;
    nh->param<bool>("viz", viz, true);
    nh->param<bool>("build_environment", build_environment, true);

    nh->param<double>("tracker_start_time", tracker_start_time, 5);


    //Camera params
    nh->param<float>("fx", fx, 542.461710f);
    nh->param<float>("fy", fy, 543.536535f);
    nh->param<float>("cx", cx, 311.081384f);
    nh->param<float>("cy", cy, 236.535761f);
    nh->param<float>("cols", cols, 640);
    nh->param<float>("rows", rows, 480);
    nh->param<float>("depth_scale_factor", depth_scale_factor, 0.0002);

    inverse_kcam = Eigen::Matrix3f::Identity();
    inverse_kcam(0, 0) = 1.f / fx;
    inverse_kcam(1, 1) = 1.f / fy;
    inverse_kcam(0, 2) = cx * (-1.f / fx);
    inverse_kcam(1, 2) = cy * (-1.f / fy);

    //General parameters
    slamdunk::SlamDunkParams sd_params;
    sd_params.cores = cores;

    nh->param<int>("rings", (int&) sd_params.rba_rings, 3);
    nh->param<float>("kf_overlapping", sd_params.kf_overlapping, 0.8f);
    nh->param<bool>("loop_inference", sd_params.try_loop_inference, false);
    nh->param<bool>("doicp", sd_params.doicp, false);
    nh->param<float>("icp_distance", sd_params.icp_distance_th, 0.2f);
    nh->param<float>("icp_normal", sd_params.icp_normal_th, 30.0f);
    nh->param<bool>("verbose", sd_params.verbose, false);

    //Feature tracker parameters
    slamdunk::FeatureTrackerParams ft_params(cores);
    std::string feature_str;
    nh->param<std::string>("features", feature_str, "surfext");
    create_feature_tracker(ft_params, feature_str);
    nh->param<double>("winl", ft_params.active_win_length, 6.0f);
    nh->param<bool>("feat_redux", ft_params.frustum_feature_reduction, false);
    sd_params.tracker.reset(new slamdunk::FeatureTracker(inverse_kcam, cols, rows, ft_params));

    //SlamDunk
    slam_dunk.reset(new slamdunk::SlamDunk(inverse_kcam, sd_params));
    slam_dunk_scene.setSlamDunkHandle(slam_dunk);

    //cloud publisher
    pointcloud_pub = nh->advertise<sensor_msgs::PointCloud2>("/trial", 1);


    //Topics
    std::string camera_rgb_topic, camera_depth_topic, camera_pose_topic, camera_odometry_topic;
    nh->param<std::string>("camera_rgb_topic", camera_rgb_topic, "/camera/rgb/image_raw");
    nh->param<std::string>("camera_depth_topic", camera_depth_topic, "/camera/depth/image_raw");
    nh->param<std::string>("camera_pose_topic", camera_pose_topic, "/camera/pose");
    nh->param<std::string>("camera_odometry_topic", camera_odometry_topic, "/camera/odometry");

    //Image/Depth synchronized callbacks
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproxSync;
    message_filters::Subscriber<sensor_msgs::Image> m_rgb_sub(*nh, camera_rgb_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> m_depth_sub(*nh, camera_depth_topic, 1);
    message_filters::Synchronizer<ApproxSync> sync(ApproxSync(100), m_rgb_sub, m_depth_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    //Camera pose init
    camera_odometry.header.frame_id = "base";
    camera_pose.header.frame_id = "base";

    camera_pose_pub = nh->advertise<geometry_msgs::PoseStamped>(camera_pose_topic, 1);
    camera_odometry_pub = nh->advertise<nav_msgs::Odometry>(camera_odometry_topic, 1);

    Utility::create_eigen_4x4_d(0, 0, 0, -M_PI / 2.0, 0, 0, T_0_S);
    Utility::create_eigen_4x4_d(0, 0, 0, 0, 0, 0, T_S_CAM);
    Utility::create_eigen_4x4_d(0, 0, 0, 0, 0, 0, T_CAMERA_CAMERACORR);


    // Spin & Time
    ros::Rate r(hz);
    start_time = ros::Time::now().toSec();
    current_time = ros::Time::now().toSec() - start_time;

//My parm£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
    tf::TransformBroadcaster br;
    tf::Transform transform1;
    tf::Transform transform2;
    //Map
    TF_lis = new tf_listener();
//££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££


    //viewer
    if (build_environment)
        viewer = new pcl::visualization::PCLVisualizer("viewer");
    int v1(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0, 0, 0, v1);
    int v2(2);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);

    cv::namedWindow("map", cv::WINDOW_NORMAL);
    cv::namedWindow("trajectory", cv::WINDOW_NORMAL);
    //cv::moveWindow("test", 30, 30);

    //Threads
    boost::thread slam_dunk_thread(slam_dunk_loop);
    boost::thread opt_thread(check_optimized_frames);

    // Spin
    while (nh->ok()) {

        current_time = ros::Time::now().toSec() - start_time;

        if (build_environment)
            if (viewer_to_update) {
                viewer_to_update = false;
                //                check_optimized_frames();
            }

        //Imshow££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
        if (first_frame_ready && viz) {
            cv::imshow("rgb", current_frame_rgbd.m_color_image);
            cv::imshow("depth", current_frame_rgbd.m_depth_image);

            //two_d_map.show_2D_map();
            //cv::imshow("test", image);

            map_3d.show_3D_2Dmap();
            TF_lis->visualize_cami_VO(image_trajectory);
            TF_lis->visualize_cami_VO(image_3D);

            cv::Point center = cv::Point(TF_lis->tf_camiv_vicom * stepSize_rows, TF_lis->tf_camiu_vicon * stepSize_cols);
            cv::circle(image_3D, center, 15, cv::Scalar(255, 0, 0), 3);

            //for visulize the dynamic modification of the trajectory
            map_3d.trajectory_slamdunk_cami();
            cv::imshow("map", image_3D);
            cv::imshow("trajectory", image_trajectory);

            //tf broadcaster
            Eigen::Matrix4d CT = Eigen::Matrix4d::Identity();
            CT(0, 0) = -CT(0, 0);
            CT(1, 1) = -CT(1, 1);
            //Eigen::Matrix4d cami = two_d_map.T_CT_CAMI;
            Utility::eigen_4x4_d_to_tf(CT, transform1, false);
            Utility::eigen_4x4_d_to_tf(two_d_map.T_CT_CAMI, transform2, false);
            br.sendTransform(tf::StampedTransform(transform1, ros::Time::now(), "odom", "CT"));
            br.sendTransform(tf::StampedTransform(transform2, ros::Time::now(), "CT", "cami"));

            //Separate
            map_3d.T_CT_CAMI=two_d_map.T_CT_CAMI;
            map_3d.voxel_cloud(*cloud_voxel_rviz);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud_voxel_rviz, cloud_msg);
            cloud_msg.header.frame_id = "odom";
            //PUBLISH
            pointcloud_pub.publish(cloud_msg);
        }
        //££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££ 

        //Wait key
        char c = cv::waitKey(10);
        if (c > 0)
            ROS_INFO("C: %d", c);
        if (c == 113)ros::shutdown();
        if (c == 97)check_optimized_frames();
        if (c == 115){//press the "s" for save
            
            pcl::io::savePCDFileASCII("output.pcd", *cloud_voxel_rviz);
            
        }

        ros::spinOnce();
        r.sleep();

        if (build_environment) {

            boost::mutex::scoped_lock lock(view_mutex);
            //££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££            
            viewer->removeAllShapes();
            Utility::draw_reference_frame(*viewer, T_S_CAM, 0.5, "T_S_CAM");
            Utility::draw_reference_frame(*viewer, T_CAMERA_CAMERACORR, 0.5, "T_CAMERA_CAMERACORR");
            Utility::draw_reference_frame(*viewer, two_d_map.T_CAMFIRST_CT, 0.5, "T_Centroid");
            //££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££  
            viewer->spinOnce();
        }
    }

    slam_dunk_thread.join();
    opt_thread.join();
}
