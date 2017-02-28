#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iomanip>
#include <queue>
#include <deque>

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

//LAR_TOOLS
#include "lar_tools.h"

//Slamdunk
#include <slamdunk/slam_dunk.h>
#include <slamdunk/slam_dunk.h>
#include <slamdunk_extension/SlamDunkScene.h>
#include <slamdunk_extension/cuda/CudaTsdf.h>


#include <boost/thread/thread.hpp>
#include <geometry_msgs/PoseStamped.h>


//GPU H
typedef unsigned char GRID_TYPE;
void host_integrate_depth(float* image_data, float* camera_transform, float* camera_transform_inv, int rows, int cols, float fx, float fy, float cx, float cy, float pos_trunc_dist, float neg_trunc_dist, float min_dist, float max_dist, bool revert = false);
void renderVoxelGridView(float*& image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy);
void host_render_tsdf_view(float*& image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy);
float get_grid_meter();
void clear_tsdf_grid();
void create_tsdf_grid();
void delete_tsdf_grid();
int fill_grid(GRID_TYPE value = 0, bool auto_value = true);
int get_grid_size();
int get_grid_side();
void increase_value(int index);
GRID_TYPE retrieve_value(int index);
void update_voxel_grid(float* image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy, int increase);
void retrieve_grid(GRID_TYPE* grid);

//slamdunk_cuda::CudaTsdf cudaTsdf;

//CudaTsdf a;


//defines
typedef pcl::PointXYZRGB PointType;
using namespace std;

ros::NodeHandle* nh;

//Topics
ros::Publisher camera_pose_pub;
ros::Publisher camera_odometry_pub;
double current_time, start_time;
double tracker_start_time;

//Clouds&Viewer
pcl::visualization::PCLVisualizer* viewer;

//SlamDunk
boost::shared_ptr<slamdunk::SlamDunk> slam_dunk;
slamdunk::SlamDunkScene slam_dunk_scene(slam_dunk);

//Transforms
Eigen::Matrix4d T_0_S;
Eigen::Matrix4d T_S_CAM;
Eigen::Matrix4d T_0_CAM;

//Messages
geometry_msgs::PoseStamped camera_pose;
nav_msgs::Odometry camera_odometry;

//Params
float fx, fy, cx, cy, cols, rows, depth_scale_factor;
bool build_environment = false;
bool viewer_to_update = false;
bool deintegration = false;
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
std::queue<slamdunk::RGBDFrame> buffer_queue;

struct OptimizationData {
    std::string key;
    Eigen::Isometry3d old_pose;
    Eigen::Isometry3d new_pose;
    bool replace;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
std::queue<OptimizationData> optimization_queue;

//Locks
boost::mutex buffer_lock;
boost::mutex ktu_mutex;
boost::mutex view_mutex;
boost::mutex optimization_mutex;


Eigen::Isometry3d camera_view = Eigen::Isometry3d::Identity();

/**
 * Scene representation. Collects Frames RGB-D and poses;
 */
struct Scene {
    std::vector<slamdunk::RGBDFrame> frames;
    //    std::vector<pcl::PointCloud<PointType>::Ptr> clouds;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > poses;

    std::map<std::string, pcl::PointCloud<PointType>::Ptr> clouds;

    void addCloud(std::string id, pcl::PointCloud<PointType>::Ptr& cloud) {
        clouds[id] = cloud;
    }
};
Scene current_scene;

void update_cloud_view(slamdunk::SlamDunkScene& slam_dunk_scene, std::string key) {

    if (slam_dunk_scene.existsCloud(key)) {
        boost::mutex::scoped_lock vis_lock(view_mutex);
        viewer->removePointCloud(key);
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(*(slam_dunk_scene.clouds[key]), *cloud, slam_dunk_scene.poses[key].matrix());
        viewer->addPointCloud(cloud, key);
    }
}

void update_cloud_view(std::string key, Eigen::Isometry3d& pose) {

    if (slam_dunk_scene.existsCloud(key)) {
        boost::mutex::scoped_lock vis_lock(view_mutex);
        viewer->removePointCloud(key);
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(*(slam_dunk_scene.clouds[key]), *cloud, pose.matrix());
        viewer->addPointCloud(cloud, key);
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

    boost::mutex::scoped_lock lock(buffer_lock);

    current_frame_rgbd.m_timestamp = rgb->header.stamp.nsec * 10e-9;
    current_frame_rgbd.secs = depth->header.stamp.sec;
    current_frame_rgbd.nsecs = depth->header.stamp.nsec;

    //    frame_buffer[frame_buffer_index] = current_frame_rgbd;
    //    frame_buffer_time[frame_buffer_index] = BufferTime(depth->header.stamp.sec, depth->header.stamp.nsec);
    buffer_queue.push(current_frame_rgbd);
    ROS_INFO("Buffer size: %d!", (int) buffer_queue.size());
    lock.unlock();

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

void check_optimized_frames();

void integrate_frame(std::string key, Eigen::Isometry3d& pose, bool deintegrate = false) {

    double time = ros::Time::now().toSec();

    host_integrate_depth(
            (float*) slam_dunk_scene.rgbd_frames[key].m_depth_image.data,
            pose.cast<float>().matrix().data(),
            pose.cast<float>().inverse().matrix().data(),
            rows, cols, fx, fy, cx, cy, 0.1, 0.1, 0.1, 3.0, deintegrate);
    time = ros::Time::now().toSec() - time;
    ROS_INFO("%s time %f", deintegrate ? "Deintegrate" : "Integrate", time * 1000.f);
}

Eigen::Isometry3d desired_pose;

void render_current_view(Eigen::Isometry3d& current_pose) {
    //                                        ROS_INFO("Tracking");

    Eigen::Matrix4f t;
    t << 0.227462, 0.604571, -0.763384, 0.869074,
            -0.452043, 0.759904, 0.467122, -0.442941,
            0.862507, 0.238829, 0.446141, 0.500151,
            0, 0, 0, 1;

    double time = ros::Time::now().toSec();
    float* image_data = NULL;
    host_render_tsdf_view(image_data,
            t.data(),
            rows, cols, fx, fy, cx, cy);
    cv::Mat ray(rows, cols, CV_32FC1, image_data);

    cv::imshow("ray", ray);
    //    cv::waitKey(10);
    if (image_data != NULL)
        delete[] image_data;
    time = ros::Time::now().toSec() - time;
    ROS_INFO("Rendering time %f", time * 1000.f);
}

/**
 * SlamDunk Loop 
 */
void slam_dunk_loop() {
    Eigen::Isometry3d current_pose;

    while (ros::ok()) {
        if (current_time > tracker_start_time && first_frame_ready) {

            boost::mutex::scoped_lock lock(buffer_lock);
            if (buffer_queue.size() <= 0)continue;
            //Consume from buffer

            //            slamdunk::RGBDFrame frame = slamdunk::cloneRGBDFrame(frame_buffer[frame_buffer_index]);
            //            BufferTime frame_time = frame_buffer_time[frame_buffer_index];
            //            frame_buffer_index++;
            //            frame_buffer_index = frame_buffer_index % frame_buffer_size;
            slamdunk::RGBDFrame frame = slamdunk::cloneRGBDFrame(buffer_queue.front());
            buffer_queue.pop();
            lock.unlock();

            //Check if Frame is void //TODO: this must be never true!
            if (frame.m_color_image.empty() || frame.m_depth_image.empty())continue;


            //Slamdunk thick!
            const int tracked = (*slam_dunk)(frame, current_pose);
            double time;
            //Update Environment
            if (build_environment) {
                if (tracked != slamdunk::SlamDunk::TRACKING_FAILED) {

                    std::string key = slam_dunk_scene.getKeyframeName(frame.m_timestamp);
                    bool is_keyframe = tracked == slamdunk::SlamDunk::KEYFRAME_DETECTED;

                    pcl::PointCloud<PointType>::Ptr cloud = slam_dunk_scene.createCloudFromFrame(frame, fx, fy, cx, cy, 4);
                    slam_dunk_scene.addCloud(key, cloud);
                    slam_dunk_scene.addRGBDFrame(key, frame);
                    slam_dunk_scene.addPose(key, current_pose, is_keyframe);
                    boost::mutex::scoped_lock lock_first(optimization_mutex);
                    {
                        //Add keyframe to pose history
                        slam_dunk_scene.addPoseToHistory(key, current_pose);

                        //Add corrected keyframes to pose history
                        slamdunk::CameraTracker::StampedPoseVector poses;
                        poses = slam_dunk->getMovedFrames();
                        std::string key;
                        for (int i = 0; i < poses.size(); i++) {
                            key = slam_dunk_scene.getKeyframeName(poses[i].first);
                            slam_dunk_scene.addPoseToHistory(key, poses[i].second);
                        }
                    }
                    lock_first.unlock();


                }

                if (tracked != slamdunk::SlamDunk::TRACKING_FAILED) {
                    //                    render_current_view(current_pose);
                    desired_pose = current_pose;
                }
            }
            //camera pose
            T_S_CAM = current_pose.matrix();

            //Camera pose publisher
            T_0_CAM = T_0_S*T_S_CAM;
            lar_tools::eigen_4x4_to_geometrypose_d(T_0_CAM, camera_pose.pose);
            camera_pose_pub.publish(camera_pose);

            //
            if (tracked == slamdunk::SlamDunk::FRAME_TRACKED) {
                camera_odometry.child_frame_id = "TRACKED";
            } else if (tracked == slamdunk::SlamDunk::KEYFRAME_DETECTED) {
                camera_odometry.child_frame_id = "KEYFRAME";
            } else if (tracked == slamdunk::SlamDunk::TRACKING_FAILED) {
                camera_odometry.child_frame_id = "FAILED";
            }
            camera_odometry.header.stamp.sec = frame.secs; // ros::Time::now().sec;
            camera_odometry.header.stamp.nsec = frame.nsecs; // ros::Time::now().nsec;
            lar_tools::eigen_4x4_to_geometrypose_d(T_0_CAM, camera_odometry.pose.pose);
            camera_odometry_pub.publish(camera_odometry);

        }
    }
}

int render_counter = 0;

void update_render() {
    using namespace slamdunk;
    std::vector<OptimizationData> optimize_vector;
    int opt_counter = 0;
    int opt_counter_max = 10;

    while (ros::ok()) {


        render_current_view(desired_pose);

        opt_counter = 0;
        optimize_vector.clear();

        boost::mutex::scoped_lock lock(optimization_mutex);
        for (slamdunk::PoseHistoryIterator it = slam_dunk_scene.keyframes_pose_history.begin();
                it != slam_dunk_scene.keyframes_pose_history.end(); ++it) {
            PoseHistory* history = it->second;
            std::string key = it->first;

            if (history->neverUpdated()) {

                //                ROS_INFO("Key %s updated for first time!", key.c_str());
                OptimizationData data;
                data.key = key;
                data.new_pose = history->last->pose;
                data.replace = false;
                optimize_vector.push_back(data);
                history->markLastAsUpdated();
            } else {
                if (history->isFine()) {
                    //                    ROS_INFO("Key %s is fine!", key.c_str());
                } else {
                    ROS_INFO("Key %s is to replace!!", key.c_str());
                    OptimizationData data;
                    data.key = key;
                    data.old_pose = history->lastUpdated->pose;
                    data.new_pose = history->last->pose;
                    data.replace = true;
                    optimize_vector.push_back(data);
                    history->markLastAsUpdated();
                    opt_counter++;
                    if (opt_counter >= opt_counter_max)break;
                }
            }

        }
        lock.unlock();

        for (int i = 0; i < optimize_vector.size(); i++) {
            if (optimize_vector[i].replace) {
                integrate_frame(optimize_vector[i].key, optimize_vector[i].old_pose, true);
                integrate_frame(optimize_vector[i].key, optimize_vector[i].new_pose, false);
            } else {
                integrate_frame(optimize_vector[i].key, optimize_vector[i].new_pose);
            }
            //            update_cloud_view(optimize_vector[i].key, optimize_vector[i].old_pose);
        }

        boost::this_thread::sleep(boost::posix_time::milliseconds(50));
    }
}


bool dragging = false;
int prev_x = -1;
int prev_y = -1;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    using namespace cv;
    if (event == EVENT_LBUTTONDOWN) {
        dragging = true;
    } else if (event == EVENT_LBUTTONUP) {
        dragging = false;
    }

    float vx = (x - prev_x)*0.01;
    float vy = (y - prev_y)*0.01;

    if (event == EVENT_MOUSEMOVE && dragging) {

        float roll, pitch = vy, yaw = -vx;
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());

        Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

        Eigen::Matrix3d rotationMatrix = q.matrix();

        camera_view.rotate(rotationMatrix);
    }

    prev_x = x;
    prev_y = y;
}

/** MAIN NODE **/
int main(int argc, char** argv) {

    // Initialize ROS
    ros::init(argc, argv, "slam_test");
    ROS_INFO("slam_test node started...");
    nh = new ros::NodeHandle("~");

    create_tsdf_grid();
    clear_tsdf_grid();

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
    nh->param<bool>("deintegration", deintegration, false);

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

    lar_tools::create_eigen_4x4_d(0, 0, 0, -M_PI / 2.0, 0, 0, T_0_S);
    lar_tools::create_eigen_4x4_d(0, 0, 0, 0, 0, 0, T_S_CAM);

    // Spin & Time
    ros::Rate r(hz);
    start_time = ros::Time::now().toSec();
    current_time = ros::Time::now().toSec() - start_time;


    //viewer
    if (build_environment)
        viewer = new pcl::visualization::PCLVisualizer("viewer");

    cv::namedWindow("ray", cv::WINDOW_NORMAL);
    cv::setMouseCallback("ray", CallBackFunc, NULL);

    //Threads
    boost::thread slam_dunk_thread(slam_dunk_loop);
    boost::thread render_thread(update_render);

    // Spin
    while (nh->ok()) {

        current_time = ros::Time::now().toSec() - start_time;

        if (build_environment)
            if (viewer_to_update) {
                viewer_to_update = false;
            }

        //Imshow
        if (first_frame_ready && viz) {
            cv::imshow("rgb", current_frame_rgbd.m_color_image);
            cv::imshow("depth", current_frame_rgbd.m_depth_image);
        }
        //Wait key
        char c = cv::waitKey(10);
        if (c > 0)
            ROS_INFO("C: %d", c);
        if (c == 113)ros::shutdown();
        if (c == 83) {
            std::cout << desired_pose.matrix() << std::endl;
        }
        if (c == 81) {
            camera_view.translate(Eigen::Vector3d(-0.1, 0, 0));
        }
        if (c == 84) {
            camera_view.translate(Eigen::Vector3d(0.0, 0.1, 0));
        }
        if (c == 82) {
            camera_view.translate(Eigen::Vector3d(0.0, -0.1, 0));
        }
        if (c == 43) {
            camera_view.translate(Eigen::Vector3d(0.0, 0, 0.1));
        }
        if (c == 45) {
            camera_view.translate(Eigen::Vector3d(0.0, 0, -0.1));
        }
        ros::spinOnce();
        r.sleep();

        if (build_environment) {
            boost::mutex::scoped_lock vis_lock(view_mutex);
            viewer->spinOnce();
        }
    }

    slam_dunk_thread.join();
    render_thread.join();
    delete_tsdf_grid();
}
