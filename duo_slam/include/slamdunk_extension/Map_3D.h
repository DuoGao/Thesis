/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Map_3D.h
 * Author: Duo Gao
 *
 * Created on 7 luglio 2016, 17.48
 */

#ifndef MAP_3D_H
#define MAP_3D_H

#include "Utility.h"

cv::Mat image_3D(Voxel_grid_cols*stepSize_cols, Voxel_grid_rows*stepSize_rows, CV_8UC3);
cv::Mat image_trajectory(Voxel_grid_cols*stepSize_cols, Voxel_grid_rows*stepSize_rows, CV_8UC3,cv::Scalar(255,255,255));

class Map_3D {
public:

    typedef struct voxel {           //just like a target box(size is dynamic),counter is how many things there
        int counter;
        voxel(){
            counter=0;
        }
    } VOXEL;

    typedef struct arrow {           //assign where is the direction of the pointer pointing
        VOXEL* voxels;
    } ARROW;

    typedef struct start_from {      //start position of the arrow
        ARROW to;
        bool full;
        double highest;
        bool is_floor;
        bool is_camera;

        start_from() {
            full = false;
            highest=0.0;
            is_floor=false;
            is_camera=false;
        }
    } START_FROM;
    
    Map_3D();
    Map_3D(const Map_3D& orig);
    virtual ~Map_3D();
    
    Eigen::Matrix4d T_CT_CAMI;
    Eigen::Matrix4d T_CT_CAMFIRST; //for visualization odom in the 3D mode
    Eigen::Matrix4d T_S_CAMFIRST;
    std::map<std::string,int> MAP_CAMI_POSE;
    std::map<int,std::string> MAP_SEQ;
    
    //dynamic allocation only
    START_FROM* ground = new START_FROM[Voxel_grid_rows * Voxel_grid_cols];
    //static allocation only
    VOXEL* grid = new VOXEL[Voxel_grid_rows * Voxel_grid_cols*Voxel_grid_higs];
    
    void add_or_minus_3D_voxel(Eigen::Matrix4d T_CT_CAMI,double x,double y,double z,int cloud_mode,int add_minus_mode);
    void compute_3D_voxel_cloud(Eigen::Matrix4d T_CT_CAMI,pcl::PointCloud<PointType> &cloud_obs,int mode);
    void show_3D_2Dmap();
    void update_heightest(int index_2D);
    void voxel_cloud(pcl::PointCloud<PointType> &cloud_in);
    void zoom_img(Eigen::Matrix4d T_CT_CAMI);
    void integrate_clouds(Eigen::Isometry3d pose,pcl::PointCloud<PointType>& cloud_obs,pcl::PointCloud<PointType>& cloud_ground,int mode);
    void update_trajectory(Eigen::Isometry3d pose,std::string key);
    void trajectory_slamdunk_cami();
    
private:
    
    int compute_cami_index(Eigen::Matrix4d T);
    void HSVtoRGB( double &r, double &g, double &b, double h, double s, double v );
    
};

#endif /* MAP_3D_H */

