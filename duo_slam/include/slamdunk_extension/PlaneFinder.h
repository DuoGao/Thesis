/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   PlaneFinder.h
 * Author: daniele
 *
 * Created on June 21, 2016, 10:37 PM
 */

#ifndef PLANEFINDER_H
#define PLANEFINDER_H

#include <string>
#include <map>
#include <vector>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

namespace slamdunk {

    template <typename PointT> class PlaneFinder {
    public:



        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;
        typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        
        /**
         * This Class represents a Plane with own attributes
         */
        struct SimplePlane {
            std::string name;
            PointCloudConstPtr input_cloud;
            std::vector<int> planes_indices;
            std::vector<int> rest_indices;
            Eigen::Vector3d plane_normal;
            pcl::ModelCoefficients plane_coefficients;
            bool valid;
            
            SimplePlane(){
                this->valid = false;
            }
            SimplePlane(PointCloudPtr& input_cloud) {
                this->input_cloud = input_cloud;
            }

            void build(int min_inliers) {
                this->plane_normal(0) = plane_coefficients.values[0];
                this->plane_normal(1) = plane_coefficients.values[1];
                this->plane_normal(2) = plane_coefficients.values[2];

                this->valid = this->planes_indices.size() >= min_inliers;
            }
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        PlaneFinder(PointCloudPtr &cloud,int min_inliers = 10000, int max_iterations = 100, float distance_th = 0.03, bool optimize_coefficient = true);
        PlaneFinder(const PlaneFinder& orig);
        virtual ~PlaneFinder();

        PointCloudPtr input_cloud;

        static void segmentMajorPlane(PointCloudPtr& cloud, std::vector<int>& indices, SimplePlane& plane, int min_inliers = 100, int max_iterations = 1000, float distance_th = 0.03, bool optimize_coefficient = true);
        std::vector<SimplePlane> planes;
    protected:

    };
}
#endif /* PLANEFINDER_H */
