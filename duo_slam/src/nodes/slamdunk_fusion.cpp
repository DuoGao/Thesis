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
#include "lar_vision/reconstruction/tsdf/tsdf_volume_octree.h"

//Slamdunk
#include <slamdunk/slam_dunk.h>
#include <slamdunk/slam_dunk.h>

#include <boost/thread/thread.hpp>
#include <geometry_msgs/PoseStamped.h>


//defines
typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
using namespace std;

ros::NodeHandle* nh;

//Topics
ros::Publisher camera_pose_pub;
ros::Publisher camera_odometry_pub;
double current_time, start_time;
double tracker_start_time;

//Clouds&Viewer
pcl::visualization::PCLVisualizer* viewer;
lar_vision::TSDFVolumeOctree::Ptr tsdf;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr raytraced(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
pcl::PointCloud<PointType>::Ptr raytraced_simple(new pcl::PointCloud<PointType>());
cv::Mat raytraced_image(640, 480, CV_8UC3);
vector<lar_vision::RGBNode::Ptr> tsdf_nodes_out;
pcl::PointCloud<PointType>::Ptr tsdf_cloud(new pcl::PointCloud<PointType>());
//float resolution = 0.1f;
//pcl::octree::OctreePointCloudSinglePoint<PointType>octree(resolution);
//

struct VoxelGrid {
    double* grid;
    double side;
    double resolution;
    int size;
    int full_size;
    std::vector<int> updated_indices;

    VoxelGrid(double side, double resolution) {
        this->side = side;
        this->resolution = resolution;
        this->size = this->side / this->resolution;
        this->full_size = this->size * this->size * this->size;
        grid = new double[this->full_size];
        this->reset();
    }

    void reset() {
        memset(grid, 0, sizeof (this->full_size));
        updated_indices.clear();
    }

    void indexToCoordinate(int index, double& x, double& y, double& z) {
        z = index / (size * size);
        y = (index % (size * size)) / size;
        x = (index % (size * size)) % size;
    }

    int indexByCoordinates(double x, double y, double z) {
        int ix = size * x / side;
        int iy = size * y / side;
        int iz = size * z / side;
        return ix + iy * size + iz * size*size;
    }

    void set(double x, double y, double z, double value) {
        int index = indexByCoordinates(x, y, z);
        if (index < this->full_size && index > 0) {
            grid[index] = value;
            updated_indices.push_back(index);
        }
    }

    void integrate(double x, double y, double z, double value) {
        int index = indexByCoordinates(x, y, z);
        //        ROS_INFO("Integrating on %d",index);
        if (index < this->full_size && index > 0) {
            grid[index] += value;
            updated_indices.push_back(index);
        }
    }

};
VoxelGrid voxel_grid(1.0, 0.02);

//SlamDunk
boost::shared_ptr<slamdunk::SlamDunk> slam_dunk;

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
bool raytrace_update = false;
int raytrace_reduction = 1;

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

//Locks
boost::mutex buffer_lock;
boost::condition_variable buffer_lock_variable;
boost::mutex tsdf_lock;
boost::condition_variable tsdf_lock_variable;

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

/**
 * Create a 3D Cloud from an RGB-D Frame
 */
pcl::PointCloud<PointType>::Ptr create_cloud_from_frame(slamdunk::RGBDFrame& frame, int sample_jumps = 1) {
    sample_jumps = sample_jumps > 1 ? sample_jumps : 1;

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    cloud->width = frame.m_color_image.cols;
    cloud->height = frame.m_color_image.rows;
    cv::Mat rgb = frame.m_color_image;
    cv::Mat depth = frame.m_depth_image;
    cv::Vec3b val;
    float d;
    const float bad_point = std::numeric_limits<float>::quiet_NaN();
    for (float y = 0; y < depth.rows; y += sample_jumps) {
        for (float x = 0; x < depth.cols; x += sample_jumps) {
            d = depth.at<float>(y, x);
            PointType p;
            p.x = (d / fx)*(x - cx);
            p.y = (d / fy)*(y - cy);
            p.z = d;
            val = rgb.at<cv::Vec3b>(y, x);
            p.r = val[2];
            p.g = val[1];
            p.b = val[0];
            cloud->points.push_back(p);
        }

    }


    return cloud;
}

void RGBtoHSV(double r, double g, double b, double *h, double *s, double *v) {
    double min, max, delta;
    min = r < g ? r : g;
    min = b < min ? b : min;
    
    max = r > g ? r : g;
    max = b > max ? b : max;
    
    *v = max; // v
    delta = max - min;
    if (max != 0)
        *s = delta / max; // s
    else {
        // r = g = b = 0		// s = 0, v is undefined
        *s = 0;
        *h = -1;
        return;
    }
    if (r == max)
        *h = (g - b) / delta; // between yellow & magenta
    else if (g == max)
        *h = 2 + (b - r) / delta; // between cyan & yellow
    else
        *h = 4 + (r - g) / delta; // between magenta & cyan
    *h *= 60; // degrees
    if (*h < 0)
        *h += 360;
}

void HSVtoRGB(double *r, double *g, double *b, double h, double s, double v) {
    int i;
    double f, p, q, t;
    if (s == 0) {
        // achromatic (grey)
        *r = *g = *b = v;
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
            *r = v;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = v;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = v;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = v;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = v;
            break;
        default: // case 5:
            *r = v;
            *g = p;
            *b = q;
            break;
    }
}

cv::Mat image_from_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& raytraced, Eigen::Isometry3d& camera_pose) {
    //    ROS_INFO("%d x %d", raytraced->width, raytraced->height);
    cv::Mat image(raytraced->height, raytraced->width, CV_8UC3);
    Eigen::Matrix4d transform = camera_pose.matrix();
    Eigen::Vector3d camera_z(
            transform(0, 2), transform(1, 2), transform(2, 2)
            );
    Eigen::Vector3d light(
            0, -1, 0
            );

    Eigen::Vector3d normal;
    Eigen::Vector3d reflection;
    double r, g, b, h, s, v;
    int index = 0;
    int w = raytraced->width;
    pcl::PointXYZRGBNormal previous_valid_point;
    for (int i = 0; i < raytraced->height; i++) {
        for (int j = 0; j < raytraced->width; j++) {
            //            ROS_INFO("W: %d, I:%d, J:%d",w,i,j);
            index = i * w + j;
            pcl::PointXYZRGBNormal& point = raytraced->points[index];
            //            while (!pcl::isFinite(point) && index < raytraced->points.size()) {
            //                point = raytraced->points[++index];
            //            }

            normal(0) = point.normal_x;
            normal(1) = point.normal_y;
            normal(2) = point.normal_z;

            double co = 2 * (light.dot(normal));
            reflection = light - co * normal;
            reflection = -reflection / normal.norm();

            double att = -(camera_z.dot(reflection) + 1) / 2.0;
            //            att = att*att;
            RGBtoHSV(128,128,128, &h, &s, &v);
            
            HSVtoRGB(&r, &g, &b, 0, 0, att*100);

            if (pcl::isFinite(point)) {

                image.at<cv::Vec3b>(i, j)[0] = b;
                image.at<cv::Vec3b>(i, j)[1] = g;
                image.at<cv::Vec3b>(i, j)[2] = r;
               
            } else {
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 0;
                image.at<cv::Vec3b>(i, j)[2] = 0;
            }
            //            pixel[0] =
            //                    pixel[1] = 0; //255 * (raytraced->points[i * w + j].z / 2.0);
            //            pixel[2] = 0; //255 * (raytraced->points[i * w + j].z / 2.0);
        }
    }
    return image;
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

void check_optimized_frames();

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

            //Slamdunk thick!
            const int tracked = (*slam_dunk)(frame, current_pose);

            //Update Environment
            if (build_environment) {
                if (tracked == slamdunk::SlamDunk::KEYFRAME_DETECTED) {
                    std::stringstream debug;
                    debug << "c" << std::fixed << std::setprecision(6) << frame.m_timestamp;
                    current_scene.frames.push_back(frame);
                    pcl::PointCloud<PointType>::Ptr cloud = create_cloud_from_frame(frame, 1);
                    current_scene.addCloud(debug.str(), cloud);
                    ROS_INFO("Add cloud %s", debug.str().c_str());
                    current_scene.poses.push_back(current_pose.matrix());

                    //                    tsdf->integrateCloud(*cloud_trans, pcl::PointCloud<pcl::Normal>(), slam_dunk->getMovedFrames()[i].second);
                    tsdf->integrateCloud(*cloud, pcl::PointCloud<pcl::Normal> (), current_pose);

                    viewer_to_update = true;
                    //                    test();
                }

                if (tracked == slamdunk::SlamDunk::FRAME_TRACKED) {

                    pcl::PointCloud<PointType>::Ptr cloud = create_cloud_from_frame(frame, 1);
                    //                    boost::lock_guard<boost::mutex> lock(tsdf_lock);
                    tsdf->integrateCloud(*cloud, pcl::PointCloud<pcl::Normal>(), current_pose);


                    if (!tsdf->isEmpty()) {

                        raytraced = tsdf->renderColoredView(current_pose, raytrace_reduction);
                        //                        tsdf_lock.unlock();

                        if (!raytrace_update) {
                            raytraced_image = image_from_cloud(raytraced, current_pose);
                            raytrace_update = true;
                            cv::imshow("ray", raytraced_image);
                            raytrace_update = false;
                        }
                        //                        ROS_INFO("Raytraced %d", raytraced->points.size());
                    }
                    tsdf_lock.unlock();

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
            camera_odometry.header.stamp.sec = frame_time.secs; // ros::Time::now().sec;
            camera_odometry.header.stamp.nsec = frame_time.nsecs; // ros::Time::now().nsec;
            lar_tools::eigen_4x4_to_geometrypose_d(T_0_CAM, camera_odometry.pose.pose);
            camera_odometry_pub.publish(camera_odometry);

            //            ROS_INFO("Tracking result: %d", tracked);

            //boost::this_thread::sleep_for(boost::chrono::milliseconds(10));

        }
    }
}

void check_optimized_frames() {
    if (!build_environment)return;

    viewer->removeAllPointClouds();
    std::stringstream ss;
    Eigen::Matrix4d frame_t;



    //    boost::lock_guard<boost::mutex> lock(tsdf_lock);
    //    tsdf->reset();
    //    tsdf_lock.unlock();

    //    voxel_grid.reset();
    //    pcl::PointCloud<PointType>::Ptr voxel_cloud(new pcl::PointCloud<PointType>());
    //
    //    slamdunk::CameraTracker::StampedPoseVector poses;
    //    //slam_dunk->getMappedPoses(poses);
    //    poses = slam_dunk->getMovedFrames();
    //
    //    std::stringstream debug;
    //    for (int i = 0; i < poses.size(); i++) {
    //        debug.str("");
    //        debug << "c" << std::fixed << std::setprecision(6) << poses[i].first;
    //        ROS_INFO("Searching for: %s", debug.str().c_str());
    //        if (current_scene.clouds.find(debug.str()) != current_scene.clouds.end()) {
    //            pcl::PointCloud<PointType>::Ptr cloud = current_scene.clouds[debug.str()];
    //            pcl::PointCloud<PointType>::Ptr cloud_trans = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    //            frame_t = poses[i].second.matrix();
    //            pcl::transformPointCloud(*cloud, *cloud_trans, frame_t);
    //
    //            //            boost::lock_guard<boost::mutex> lock(tsdf_lock);
    //            tsdf->integrateCloud(*cloud, pcl::PointCloud<pcl::Normal>(), poses[i].second);
    //            //            tsdf_lock.unlock();
    //            //            tsdf->integrateCloud(*cloud, pcl::PointCloud<pcl::Normal>(), slam_dunk->getMovedFrames()[i].second);
    //            //            for (int j = 0; j < cloud_trans->size(); j++) {
    //            //                PointType& p = cloud_trans->points[j];
    //            //                voxel_grid.integrate(p.x, p.y, p.z, 1);
    //            //            }
    //
    //        }
    //    }


    //    if (tsdf_cloud->size() > 0)
    //        viewer->addPointCloud(tsdf_cloud, "c");
    //    ROS_INFO("POses: %s", debug.str().c_str());
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
    nh->param<int>("raytrace_reduction", raytrace_reduction, 1);


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

    //TSDF
    using namespace lar_vision;
    tsdf = TSDFVolumeOctree::Ptr(new TSDFVolumeOctree);
    tsdf->setCameraIntrinsics(fx, fy, cx, cy);

    double tsdf_size = 3;
    int tsdf_resolution = 1024;
    double tsdf_trunc_dist_pos;
    double tsdf_trunc_dist_neg;
    nh->param<double>("tsdf_size", tsdf_size, 3);
    nh->param<int>("tsdf_resolution", tsdf_resolution, 2048);
    nh->param<double>("tsdf_trunc_dist_pos", tsdf_trunc_dist_pos, 0.05);
    nh->param<double>("tsdf_trunc_dist_neg", tsdf_trunc_dist_neg, 0.05);

    tsdf->setGridSize(tsdf_size, tsdf_size, tsdf_size); // 10m x 10m x 10m
    tsdf->setResolution(tsdf_resolution, tsdf_resolution, tsdf_resolution); // Smallest cell size = 10m / 2048 = about half a centimeter
    //Eigen::Affine3d tsdf_center; // Optionally offset the center
    Eigen::Affine3d affine_tsdf_center;
    affine_tsdf_center.translation() = Eigen::Vector3d(0, 0, 50);
    tsdf->setGlobalTransform(affine_tsdf_center);
    tsdf->setImageSize(cols, rows);
    tsdf->setSensorDistanceBounds(-30.0, 30.0f);
    tsdf->setIntegrateColor(true);
    tsdf->setDepthTruncationLimits(tsdf_trunc_dist_pos, tsdf_trunc_dist_neg);
    tsdf->reset(); // Initialize it to be empty

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

    //Threads
    boost::thread slam_dunk_thread(slam_dunk_loop);

    // Spin
    while (nh->ok()) {

        current_time = ros::Time::now().toSec() - start_time;

        if (build_environment) {
            if (viewer_to_update) {
                viewer_to_update = false;
                check_optimized_frames();
            }

            //            if (raytrace_update) {
            //                cv::imshow("ray", raytraced_image);
            //                raytrace_update = false;
            //            }
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
        if (c == 97)check_optimized_frames();

        ros::spinOnce();
        r.sleep();

        if (build_environment)
            viewer->spinOnce();
    }

    slam_dunk_thread.join();

}
