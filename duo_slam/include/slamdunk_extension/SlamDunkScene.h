
#ifndef SLAMDUNKSCENE_H
#define SLAMDUNKSCENE_H

#include "slam_dunk.h"
#include <string>
#include <map>
#include <vector>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/features/integral_image_normal.h>

//Boost
#include <boost/thread/thread.hpp>

namespace slamdunk {

    typedef pcl::PointXYZRGB SlamDunkCloudType;
    typedef pcl::Normal SlamDunkNormalType;

    /**
     * Pose Entry struct representing a pose status in times 
     */
    typedef struct PoseEntry {
        PoseEntry* next;
        Eigen::Isometry3d pose;
        int index;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PoseEntry() {
            next = NULL;
            index = -1;
        }

        void del() {
            if (next != NULL) {
                next->del();
                delete next;
                next = NULL;
            }
        }

        PoseEntry* getLast() {
            if (next != NULL) {
                return next->getLast();
            } else {
                return this;
            }
        }

        void append(PoseEntry* pose) {
            this->next = pose;
            this->next->index = this->index + 1;
        }
    } PoseEntry;

    /**
     * Manages a collection of PoseEntry, keeping first/last and lastUpdated one
     */
    typedef struct PoseHistory {
        PoseEntry* first;
        PoseEntry* last;
        PoseEntry* lastUpdated;
        int size;

        PoseHistory() {
            first = last = lastUpdated = NULL;
            size = 0;
        }

        int getSize() {
            return size;
        }

        void appendPose(PoseEntry* pose) {
            if (first == NULL) {
                first = pose;
                first->index = 0;
            } else {
                first->getLast()->append(pose);
            }
            last = pose;
            size++;
        }

        void appendPose(Eigen::Isometry3d& epose) {
            PoseEntry* pose = new PoseEntry();
            pose->pose = epose;
            appendPose(pose);
        }

        bool getPoseByIndex(int index, PoseEntry*& pose) {
            pose = first;
            while (pose->index != index || pose == NULL) {
                pose = pose->next;
            }
            return pose != NULL;
        }

        bool markAsUpdatedByIndex(int index) {
            PoseEntry* pose;
            if (getPoseByIndex(index, pose)) {
                lastUpdated = pose;
                return true;
            }
            return false;
        }

        bool markLastAsUpdated() {
            if (last != NULL) {
                lastUpdated = last;
                return true;
            }
            return false;
        }

        PoseEntry* getLastUpdated() {
            return lastUpdated;
        }

        void del(PoseEntry*& p) {
            if (p != NULL) {
                p->del();
                delete p;
                p = NULL;
            }
        }

        bool neverUpdated() {
            return getLastUpdated() == NULL;
        }

        bool isFine() {
            return getLastUpdated() == last;
        }

        void del() {
            del(first);
            del(last);
            del(lastUpdated);
        }


    } PoseHistory;

    /**
     * Entry used to integrate/deintegrate frame data
     */
    typedef struct ConsumingEntry {
        std::string key;
        Eigen::Isometry3d old_pose;
        Eigen::Isometry3d new_pose;
        bool replacement;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        /**
         * New Entry
         */
        ConsumingEntry(std::string key, Eigen::Isometry3d& new_pose) {
            this->key = key;
            this->new_pose = new_pose;
            this->replacement = false;
        }

        /**
         * Replace Entry
         */
        ConsumingEntry(std::string key, Eigen::Isometry3d& old_pose, Eigen::Isometry3d& new_pose) {
            this->key = key;
            this->old_pose = old_pose;
            this->new_pose = new_pose;
            this->replacement = true;
        }
    } ConsumningEntry;

    /**
     * 
     */
    class SlamDunkScene {
    public:
        SlamDunkScene();
        SlamDunkScene(boost::shared_ptr<slamdunk::SlamDunk> slamdunk);
        virtual ~SlamDunkScene();
        static std::string getKeyframeName(double timestamp);
        static pcl::PointCloud<SlamDunkCloudType>::Ptr createCloudFromFrame(RGBDFrame& frame, double fx = 525., double fy = 525, double cx = 320, double cy = 240, int sample_jumps = 1);
        static pcl::PointCloud<SlamDunkNormalType>::Ptr computeNormals(pcl::PointCloud<SlamDunkCloudType>::Ptr& cloud);
        std::map<std::string, pcl::PointCloud<SlamDunkCloudType>::Ptr>& getClouds();
        std::map<std::string, RGBDFrame>& getRGBDFrames();
        std::map<std::string, Eigen::Isometry3d, std::less<std::string>, Eigen::aligned_allocator<std::pair<const std::string, Eigen::Isometry3d> > >& getPoses();
        void addCloud(std::string& key, pcl::PointCloud<SlamDunkCloudType>::Ptr& cloud);
        bool existsCloud(std::string& key);
        void addRGBDFrame(std::string& key, RGBDFrame& frame);
        bool existsRGBDFrame(std::string& key);
        void addPose(std::string& key, Eigen::Isometry3d& pose, bool keyframe = false);
        void addPoseToHistory(std::string& key, Eigen::Isometry3d& pose);
        bool existsPose(std::string& key);
        bool existsPoseHistory(std::string& key);

        //History management
        void setSlamDunkHandle(boost::shared_ptr<slamdunk::SlamDunk> slamdunk);
        void updatePoseHistory(std::string& key, Eigen::Isometry3d& pose, bool update_optimized_frames = true);
        void getAvailableEntries(std::vector<ConsumingEntry>& entries, int max_entries = 10, bool include_newentries_in_counter = false);

        bool addHierarchy(std::string& key, std::string& key_2);
        bool existHierarchy(std::string& key);

        std::map<std::string, pcl::PointCloud<SlamDunkCloudType>::Ptr> clouds;
        std::map<std::string, RGBDFrame> rgbd_frames;
        std::map<std::string, Eigen::Isometry3d, std::less<std::string>, Eigen::aligned_allocator<std::pair<const std::string, Eigen::Isometry3d> > > poses;
        std::map<std::string, std::vector<std::string> > hierarchy;
        std::map<std::string, bool > keyframes_mask;
        std::map<std::string, PoseHistory*> keyframes_pose_history;

        std::string last_keyframe_key;
        Eigen::Isometry3d last_keyframe_pose;
        bool isEmpty();

        bool relativeDeltaPose(std::string key_parent, std::string key_child, Eigen::Isometry3d& delta_pose);
    protected:
        boost::mutex history_lock;
        boost::shared_ptr<slamdunk::SlamDunk> _slamdunk;
    };


    typedef std::map<std::string, PoseHistory*>::iterator PoseHistoryIterator;
}
#endif /* SLAMDUNKSCENE_H */

