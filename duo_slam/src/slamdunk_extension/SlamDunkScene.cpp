
#include "SlamDunkScene.h"

namespace slamdunk {

    SlamDunkScene::SlamDunkScene() {
        this->_slamdunk = NULL;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    SlamDunkScene::SlamDunkScene(boost::shared_ptr<slamdunk::SlamDunk> slamdunk) {
        this->_slamdunk = slamdunk;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    SlamDunkScene::~SlamDunkScene() {
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::setSlamDunkHandle(boost::shared_ptr<slamdunk::SlamDunk> slamdunk) {
        this->_slamdunk = slamdunk;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    std::string SlamDunkScene::getKeyframeName(double timestamp) {
        std::stringstream ss;
        ss << "c" << std::fixed << std::setprecision(6) << timestamp;
        return ss.str();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    pcl::PointCloud<SlamDunkCloudType>::Ptr SlamDunkScene::createCloudFromFrame(RGBDFrame& frame, double fx, double fy, double cx, double cy, int sample_jumps) {
        sample_jumps = sample_jumps > 1 ? sample_jumps : 1;

        pcl::PointCloud<SlamDunkCloudType>::Ptr cloud(new pcl::PointCloud<SlamDunkCloudType>);
        cloud->width = frame.m_color_image.cols / sample_jumps;
        cloud->height = frame.m_color_image.rows / sample_jumps;
        cv::Mat rgb = frame.m_color_image;
        cv::Mat depth = frame.m_depth_image;
        cv::Vec3b val;
        float d;
        const float bad_point = std::numeric_limits<float>::quiet_NaN();
        for (float y = 0; y < depth.rows; y += sample_jumps) {
            for (float x = 0; x < depth.cols; x += sample_jumps) {
                d = depth.at<float>(y, x);
                SlamDunkCloudType p;
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
    /////////////////////////////////////////////////////////////////////////////////////////////////

    pcl::PointCloud<SlamDunkNormalType>::Ptr SlamDunkScene::computeNormals(pcl::PointCloud<SlamDunkCloudType>::Ptr& cloud) {
        pcl::IntegralImageNormalEstimation<SlamDunkCloudType, SlamDunkNormalType> ne;
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.02f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setInputCloud(cloud);
        pcl::PointCloud<SlamDunkNormalType>::Ptr normals(new pcl::PointCloud<SlamDunkNormalType>);
        ne.compute(*normals);
        return normals;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::existsCloud(std::string& key) {
        return clouds.find(key) != clouds.end();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::existsPose(std::string& key) {
        return poses.find(key) != poses.end();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::existsPoseHistory(std::string& key) {
        return keyframes_pose_history.find(key) != keyframes_pose_history.end();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::existsRGBDFrame(std::string& key) {
        return rgbd_frames.find(key) != rgbd_frames.end();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::addCloud(std::string& key, pcl::PointCloud<SlamDunkCloudType>::Ptr& cloud) {
        clouds[key] = cloud;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::addPose(std::string& key, Eigen::Isometry3d& pose, bool keyframe) {
        poses[key] = pose;
        if (keyframe) {
            last_keyframe_key = key;
            last_keyframe_pose = pose;
            keyframes_mask[key] = true;


        }

    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::addPoseToHistory(std::string& key, Eigen::Isometry3d& pose) {
        //pose history
        if (!existsPoseHistory(key)) {
            keyframes_pose_history[key] = new PoseHistory();
        }
        keyframes_pose_history[key]->appendPose(pose);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::addRGBDFrame(std::string& key, RGBDFrame& frame) {
        rgbd_frames[key] = frame;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::addHierarchy(std::string& key, std::string& key_2) {
        if (existHierarchy(key)) {
            hierarchy[key].push_back(key_2);
        } else {
            hierarchy[key] = std::vector<std::string> ();
            hierarchy[key].push_back(key_2);
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::existHierarchy(std::string& key) {
        return hierarchy.find(key) != hierarchy.end();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::isEmpty() {
        return poses.size() <= 0;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////

    bool SlamDunkScene::relativeDeltaPose(std::string key_parent, std::string key_child, Eigen::Isometry3d& delta_pose) {
        if (existsPose(key_parent) && existsPose(key_child)) {
            Eigen::Isometry3d pose_parent = poses[key_parent];
            Eigen::Isometry3d pose_child = poses[key_child];
            pose_parent = pose_parent.inverse();
            delta_pose = pose_parent*pose_child;
            return true;
        }
        return false;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::updatePoseHistory(std::string& key, Eigen::Isometry3d& pose, bool update_optimized_frames) {

        //Refactoring
        //        assert(_slamdunk);

        boost::mutex::scoped_lock lock(history_lock);

        //Add keyframe to pose history
        addPoseToHistory(key, pose);

        //Add corrected keyframes to pose history
        if (update_optimized_frames) {
            slamdunk::CameraTracker::StampedPoseVector poses;
            poses = _slamdunk->getMovedFrames();
            std::string key;
            for (int i = 0; i < poses.size(); i++) {
                key = getKeyframeName(poses[i].first);
                addPoseToHistory(key, poses[i].second);
            }
        }

        lock.unlock();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    void SlamDunkScene::getAvailableEntries(std::vector<ConsumingEntry>& entries, int max_entries, bool include_newentries_in_counter) {
        //Refactoring
        assert(_slamdunk);

        boost::mutex::scoped_lock lock(history_lock);
        entries.clear();

        int opt_counter = 0;
        for (slamdunk::PoseHistoryIterator it = keyframes_pose_history.begin();
                it != keyframes_pose_history.end(); ++it) {
            PoseHistory* history = it->second;
            std::string key = it->first;

            if (history->neverUpdated()) {

                entries.push_back(ConsumingEntry(key, history->last->pose));
                history->markLastAsUpdated();
                if (include_newentries_in_counter) {
                    opt_counter++;
                }
            } else {
                if (history->isFine()) {
                    //Nothing to do
                } else {
                    entries.push_back(ConsumingEntry(key, history->lastUpdated->pose, history->last->pose));
                    history->markLastAsUpdated();
                    opt_counter++;

                }
            }
            //Max reached
            if (opt_counter >= max_entries)break;
        }

        lock.unlock();
    }

}
