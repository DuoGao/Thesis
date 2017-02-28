/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Map_3D_new.h
 * Author: Duo Gao
 *
 * Created on August 1, 2016, 6:05 PM
 */

#ifndef MAP_3D_NEW_H
#define MAP_3D_NEW_H

#include "Utility.h"

cv::Mat image_3D_new(Voxel_grid_cols*stepSize_cols, Voxel_grid_rows*stepSize_rows, CV_8UC3);
cv::Mat image_trajectory_new(Voxel_grid_cols*stepSize_cols, Voxel_grid_rows*stepSize_rows, CV_8UC3,cv::Scalar(255,255,255));

class Map_3D_new {
public:

    typedef struct voxel { 
        int counter;
        voxel() {
            counter = 0;
        }
    } VOXEL;

    typedef struct voxel_block {
        voxel_block* next;
        voxel* voxels;
        int index;
        //the size of big box can be calculate automatically inside 
        float voxel_block_meter_size;

        voxel_block() {
        }

        voxel_block(int i, int size) {
            voxels = new voxel[size];
            index = i;
            next = NULL;
            voxel_block_meter_size=(float)unit_height*size;
        }

        float bottomSurfce() {
            return index * voxel_block_meter_size;
        }

        float topSurfce() {
            return (index + 1) * voxel_block_meter_size;
        }

    } VOXEL_BLOCK;

    typedef struct start_from { //start position of the arrow
        voxel_block* VOXEL_BLOCKS;
        bool full;
        double highest;
        bool is_floor;
        bool is_camera;

        start_from() {
            full = false;
            highest = 0.0;
            is_floor = false;
            is_camera = false;
        }
    } START_FROM;
    
    Map_3D_new();
    Map_3D_new(const Map_3D_new& orig);
    virtual ~Map_3D_new();
    
    Eigen::Matrix4d T_CAMFIRST_CAMI;
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
    
    void voxel_cloud(pcl::PointCloud<PointType> &cloud_in);
    void zoom_img(Eigen::Matrix4d T_CT_CAMI);
    void integrate_clouds(Eigen::Isometry3d pose,pcl::PointCloud<PointType>& cloud_obs,pcl::PointCloud<PointType>& cloud_ground,int mode);
    void update_trajectory(Eigen::Isometry3d pose,std::string key);
    void trajectory_slamdunk_cami();

    //for linked list
    void update_heightest(voxel_block *head);
    void addNode(voxel_block *head,int voxel_seq,int mode);
    void insertFront(voxel_block **head,int voxel_seq);
    
    int in_which_list(int J);
    int local_index(int voxel_seq,voxel_block *head);
    std::string check_linked_list(voxel_block *head,int voxel_seq,int mode);
private:
    
    int compute_cami_index(Eigen::Matrix4d T);
    void HSVtoRGB( double &r, double &g, double &b, double h, double s, double v );
};

#endif /* MAP_3D_NEW_H */

