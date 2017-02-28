/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   PlaneFinder.cpp
 * Author: daniele
 * 
 * Created on June 21, 2016, 10:37 PM
 */

#include "PlaneFinder.h"
namespace slamdunk {

    template <typename PointT>
    PlaneFinder<PointT>::PlaneFinder(PointCloudPtr &cloud, int min_inliers, int max_iterations, float distance_th, bool optimize_coefficient) {
        this->input_cloud = cloud;


        std::vector<int> current_plane_indices;
        for (int i = 0; i < cloud->size(); i+=4) {
            current_plane_indices.push_back(i);
        }

        bool finish = false;
        do {
            SimplePlane plane;
            segmentMajorPlane(cloud, current_plane_indices, plane, min_inliers, max_iterations, distance_th, optimize_coefficient);
            if (!plane.valid || plane.planes_indices.size() <= 0) {
                finish = true;
            } else {
                planes.push_back(plane);
                current_plane_indices = plane.rest_indices;
                printf("Current size reduced to: %d \n", (int) current_plane_indices.size());
                //                finish=true;
            }
        } while (!finish);
    }

    template <typename PointT>
    PlaneFinder<PointT>::PlaneFinder(const PlaneFinder& orig) {
    }

    template <typename PointT>
    PlaneFinder<PointT>::~PlaneFinder() {
    }

    template<typename PointT>
    void PlaneFinder<PointT>::segmentMajorPlane(PointCloudPtr& cloud, std::vector<int>& indices, SimplePlane& plane, int min_inliers, int max_iterations, float distance_th, bool optimize_coefficient) {

        //Create the SimplePLane object
        plane = SimplePlane(cloud);

        //Segmentation 
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(optimize_coefficient);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(max_iterations);
        seg.setDistanceThreshold(distance_th);

        PointCloudPtr cloud_c(new PointCloud);
//        printf("Cloud created\n");
        if (indices.size() > 0) {
            //            pcl::IndicesPtr ind(new std::vector<int>);
            //            *ind = indices;
            //            ind->insert(ind->end(), indices.begin(), indices.end());
            //            printf("Searching on %d / %d\n", seg.getIndices()->size(), cloud->size());
            pcl::copyPointCloud(*cloud, indices, *cloud_c);
        } else {
            cloud_c = cloud;
//            printf("Searching whole cloud");
        }
        seg.setInputCloud(cloud_c);
        seg.segment(*inliers, plane.plane_coefficients);

        std::vector<int> p_indices;
        std::vector<int> r_indices;
        // Extract the inliers of PLANES and for REST
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_c);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(p_indices);
        extract.setNegative(true);
        extract.filter(r_indices);

        for (int i = 0; i < p_indices.size(); i++) {

//            printf("Try indices P %d / %d\n", p_indices[i], indices.size());
            if (indices.size() > 0)
                plane.planes_indices.push_back(indices[p_indices[i]]);
            else
                plane.planes_indices.push_back(p_indices[i]);
        }
        for (int i = 0; i < r_indices.size(); i++) {
//            printf("Try indices R %d / %d\n", r_indices[i], indices.size());
            if (indices.size() > 0)
                plane.rest_indices.push_back(indices[r_indices[i]]);
            else
                plane.rest_indices.push_back(r_indices[i]);
        }

        //Build the plane
        plane.build(min_inliers);

    }

    template class PlaneFinder<pcl::PointXYZRGB>;
}

