/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Two_D_Map.h
 * Author: Duo Gao
 *
 * Created on 24 giugno 2016, 10.51
 */

#ifndef TWO_D_MAP_H
#define TWO_D_MAP_H

#include "Utility.h"

cv::Mat image_2D(Voxel_grid_cols*stepSize_cols, Voxel_grid_rows*stepSize_rows, CV_8UC3);

class Two_D_Map {

public:
    //shared param
    Eigen::Matrix4d T_CAMFIRST_CT;
    Eigen::Matrix4d T_S_CAMFIRST;
    Eigen::Matrix4d T_S_CAMI;
    Eigen::Matrix4d T_CAMFIRST_CAMI;
    Eigen::Matrix4d T_CT_CAMI;
    Eigen::Vector3d FIRST_PLANE_NORMAL;
    pcl::ModelCoefficients FIRST_PLANE_COEFFICIENTS;
    

    struct Voxel2D {
        double height;
        bool occupied;
    };

    Voxel2D Voxel_grid[Voxel_grid_rows*Voxel_grid_cols];
    Voxel2D Voxel_grid_ground[Voxel_grid_rows*Voxel_grid_cols];
    Voxel2D Voxel_grid_camera[Voxel_grid_rows*Voxel_grid_cols];

    std::string FIRST_PLANE_NAME;
    std::stringstream CLOUD_NAME;
    std::map<std::string, pcl::PointCloud<PointType>> MAP_PLANE;
    std::map<std::string, pcl::PointCloud<PointType>> MAP_REST;

    Two_D_Map();
    Two_D_Map(const Two_D_Map& orig);
    virtual ~Two_D_Map();

    struct Cloudinfo {
        std::string name;
        pcl::PointCloud<PointType>::Ptr input_cloud;
        pcl::PointCloud<PointType>::Ptr plane_cloud;
        pcl::PointCloud<PointType>::Ptr rest_cloud;
        pcl::PointIndices plane_indices;
        pcl::PointIndices rest_indices;
        Eigen::Vector3d plane_normal;
        pcl::ModelCoefficients plane_coefficients;
        bool valid;

        Cloudinfo() {
            this->valid = false;
        }

        Cloudinfo(pcl::PointCloud<PointType>::Ptr& input_cloud) {
            this->input_cloud = input_cloud;
        }

        void build_normal(int min_points) {
            plane_normal = Eigen::Vector3d(
                    plane_coefficients.values[0],
                    plane_coefficients.values[1],
                    plane_coefficients.values[2]
                    );

            this->valid = this->plane_indices.indices.size() >= min_points;

        }

    };
    
    
    void init_Voxel_grid();
    void first_callback_init(pcl::PointCloud<PointType>::Ptr &cloud);
    void ground_finder(pcl::PointCloud<PointType>::Ptr& cloud,std::string key,Eigen::Matrix4d T_S_CAM,double Angle_limit,double Height_limit,double Area_limit);
    
    //the filters
    void pass_through_fillter(pcl::PointCloud<PointType> &input_cloud,double max,double min,bool setnegative);
    void map_cloud_fillter(pcl::PointCloud<PointType> &input_cloud,double voxel_leaf);
    void RadiusOutlierRemoval(pcl::PointCloud<PointType> &input_cloud,double radius,double min_neighborhood);
     
    void project_to_1stplane(pcl::PointCloud<PointType> &cloud_obs);
    void compute_Voxel_grid(pcl::PointCloud<PointType> &cloud_obs,int mode);
    void show_2D_map();
    
    void create_unit_cloud(pcl::PointCloud<PointType>::Ptr &cloud_unit,double unit_size,int point_num_oneedge);//just for fun
    void project_unitcloud_pcl(pcl::PointCloud<PointType>::Ptr &cloud_unit,double unit_size,double real_U,double real_V,int switch_case_phase);//just for fun


    
    
private:
    
    double Round_planez(double Height,int precision);
    bool segment_Plane(pcl::PointCloud<PointType>::Ptr& cloud, Cloudinfo& plane, int min_inliers = 10, int max_iterations = 100, float distance_th = 0.03, bool optimize_coefficient = true);
    void project_1stnormal_ref(Cloudinfo &plane_cloud,Eigen::Matrix4d &T);
    
   
    double max_of_three(double a,double b,double c);
    double min_of_three(double a,double b,double c);
    void fun_color(double height,double max_height,Eigen::Vector3d &bool_vector);
    void RGBtoHSV( double r, double g, double b, double &h, double &s, double &v );
    void HSVtoRGB( double &r, double &g, double &b, double h, double s, double v );

};

#endif /* TWO_D_MAP_H */

