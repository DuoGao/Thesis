/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Map_3D.cpp
 * Author: Duo Gao
 * 
 * Created on 7 luglio 2016, 17.48
 */



#include "Map_3D.h"

Map_3D::Map_3D() {
}

Map_3D::Map_3D(const Map_3D& orig) {
}

Map_3D::~Map_3D() {
}

int Map_3D::compute_cami_index(Eigen::Matrix4d T){
    
        //compute the index of the camera
    double ucam, vcam;
    Eigen::Vector4d camera_position = Eigen::Vector4d(
            T(0, 3),
            T(1, 3),
            T(2, 3),
            1.0);
    double c_x = camera_position[0] + Voxel_grid_cols * unit_cols / 2;
    double c_y = camera_position[1] + Voxel_grid_rows * unit_rows / 2;
    //double c_z = camera_position[2];
    modf(c_x / unit_cols, &ucam);
    modf(c_y / unit_rows, &vcam);
    int array_index_cami = (int) Voxel_grid_cols * vcam + ucam;
    
    return array_index_cami;
    
}

void Map_3D::update_heightest(int index_2D) {

    if (ground[index_2D].full == true) {

        for (int j = 0; j < Voxel_grid_higs; j++) {

            if (ground[index_2D].to.voxels[j].counter >= thre_counter) {
                ground[index_2D].highest = unit_height * (j + 1);
            }
        }
    }

}

void Map_3D::add_or_minus_3D_voxel(Eigen::Matrix4d T,double x, double y, double z,int cloud_mode,int add_minus_mode) {    

    Eigen::Vector4d point_position = Eigen::Vector4d(x, y, z, 1.0);
    double side_rows = Voxel_grid_rows*unit_rows;
    double side_cols = Voxel_grid_cols*unit_cols;
    double offset_u = side_cols / 2.0;
    double offset_v = side_rows / 2.0;

    point_position = T*point_position;
    double X = point_position[0] + offset_u;
    double Y = point_position[1] + offset_v;
    double H = point_position[2];
    double U,V,J;
    modf(X / unit_cols,&U);
    modf(Y / unit_rows, &V);
    modf(H / unit_height, &J);
    int array_index_2D = (int) Voxel_grid_cols * V + U;     
    //int array_index_3D = (int) Voxel_grid_rows_3D*Voxel_grid_cols_3D*J+Voxel_grid_cols_3D * V + U;
    int j=(int)J;
    int cami_index;

    if (cloud_mode == mode_obs) {
        if (V < Voxel_grid_rows && U < Voxel_grid_cols && J < Voxel_grid_higs)
            if (V >= 0 && U >= 0 && J >= 0) {
                //dynamic allocation,only when we need it we build the container 
                if (!ground[array_index_2D].full) {
                    ground[array_index_2D].full = true;
                    ground[array_index_2D].to.voxels = new VOXEL[Voxel_grid_higs];
                }

                if (add_minus_mode == integrate_mode) {
                    ground[array_index_2D].to.voxels[j].counter++;
                    update_heightest(array_index_2D);
                    
                    cami_index = compute_cami_index(T);
                    ground[cami_index].is_camera = true;
                }
                if (add_minus_mode == deintegrate_mode) {
                    ground[array_index_2D].to.voxels[j].counter-=1;
                    update_heightest(array_index_2D);
                    
                    cami_index = compute_cami_index(T);
                    ground[cami_index].is_camera = false;
                }
            }
    }

    if (cloud_mode == mode_plane) {
        if (V < Voxel_grid_rows && U < Voxel_grid_cols && J < Voxel_grid_higs)
            if (V >= 0 && U >= 0 && J >= 0) {
                if (!ground[array_index_2D].is_floor) {

                    if (add_minus_mode == integrate_mode) {
                        ground[array_index_2D].is_floor = true;
                        cami_index = compute_cami_index(T);
                        ground[cami_index].is_camera = true;
                    }
                    if (add_minus_mode == deintegrate_mode) {
                        ground[array_index_2D].is_floor = false;
                        cami_index = compute_cami_index(T);
                        ground[cami_index].is_camera = false;
                    }

                }

            }
    }
}

//below is the old function,do not use it but keep it for conparison of the result with optimized and unoptimized
void Map_3D::compute_3D_voxel_cloud(Eigen::Matrix4d T, pcl::PointCloud<PointType>& cloud_obs, int mode) { 

    pcl::PointCloud<PointType>::Ptr cloud_OBS(new pcl::PointCloud<PointType>);
    (*cloud_OBS) = cloud_obs;

    if (mode == mode_obs) {
        for (size_t i = 0; i < cloud_OBS->size(); ++i) {
            add_or_minus_3D_voxel(T, cloud_OBS->points[i].x, cloud_OBS->points[i].y, cloud_OBS->points[i].z, mode_obs,integrate_mode);
        }
    }

    if (mode == mode_plane) {
        for (size_t i = 0; i < cloud_OBS->size(); ++i) {
            add_or_minus_3D_voxel(T, cloud_OBS->points[i].x, cloud_OBS->points[i].y, cloud_OBS->points[i].z, mode_plane,integrate_mode); //set all the is_floor true
        }
    }
}


void Map_3D::integrate_clouds(Eigen::Isometry3d pose, pcl::PointCloud<PointType>& cloud_obs, pcl::PointCloud<PointType>& cloud_ground, int mode){
    
    pcl::PointCloud<PointType>::Ptr cloud_OBS(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_GROUND(new pcl::PointCloud<PointType>);
    (*cloud_OBS) = cloud_obs;
    (*cloud_GROUND) = cloud_ground;

    //recaculate the T_CT_CAMI
    Eigen::Matrix4d T_S_CAMI=pose.matrix();
    Eigen::Matrix4d T_CAMFIRST_CAMI=T_S_CAMFIRST.inverse() * T_S_CAMI;
    Eigen::Matrix4d T_CT_CAMI=T_CT_CAMFIRST*T_CAMFIRST_CAMI;

    //integrate or deintegrate the obstercal
    for (size_t i = 0; i < cloud_OBS->size(); ++i) {

            add_or_minus_3D_voxel(T_CT_CAMI, cloud_OBS->points[i].x, cloud_OBS->points[i].y, cloud_OBS->points[i].z, mode_obs, mode);
            
    }

    //integrate or deintegrate the floor
    for (size_t i = 0; i < cloud_GROUND->size(); ++i) {
        
            add_or_minus_3D_voxel(T_CT_CAMI, cloud_GROUND->points[i].x, cloud_GROUND->points[i].y, cloud_GROUND->points[i].z, mode_plane, mode);
    }
      
    
}


void Map_3D::update_trajectory(Eigen::Isometry3d pose,std::string key){
        
    //recaculate the T_CT_CAMI(should be in a function,for it has been called many times)
    Eigen::Matrix4d T_S_CAMI=pose.matrix();
    Eigen::Matrix4d T_CAMFIRST_CAMI=T_S_CAMFIRST.inverse() * T_S_CAMI;
    Eigen::Matrix4d T_CT_CAMI=T_CT_CAMFIRST*T_CAMFIRST_CAMI;    
    //caculate the index of camera in 2D map
    int cam_index = compute_cami_index(T_CT_CAMI);

    //using two map to keep the sequence of the in and out of the pose
    std::map<std::string, int>::iterator it;
    it = MAP_CAMI_POSE.find(key);
    
    //std::vector<std::string> keys_ordering;
    
    if (it== MAP_CAMI_POSE.end()) {
        int sequence = MAP_CAMI_POSE.size();
        MAP_SEQ[sequence]=key;               //make the sequence start from zero
        MAP_CAMI_POSE[key] = cam_index;
      //  keys_ordering.push_back(key);
    }
    else{
           
        MAP_CAMI_POSE[key] = cam_index; 
    }
}

void Map_3D::trajectory_slamdunk_cami() {

    int start, end, ucamis, vcamis, ucamie, vcamie;
    cv::Point starto;
    cv::Point endo;
    for (int u = 0; u < Voxel_grid_cols * stepSize_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows * stepSize_rows; v++) {
            if (image_trajectory.at<cv::Vec3b>(u, v)[0] == 255 && image_trajectory.at<cv::Vec3b>(u, v)[1] == 0 && image_trajectory.at<cv::Vec3b>(u, v)[2] == 255) {
                image_trajectory.at<cv::Vec3b>(u, v)[0] = 255;
                image_trajectory.at<cv::Vec3b>(u, v)[1] = 255;
                image_trajectory.at<cv::Vec3b>(u, v)[2] = 255;
            }
        }
    }
    for (int i = 0; i < MAP_CAMI_POSE.size(); i++) {
        start = MAP_CAMI_POSE[MAP_SEQ[i]];
        if (i < MAP_CAMI_POSE.size() - 1) {
            end = MAP_CAMI_POSE[MAP_SEQ[i + 1]];
            ucamis = start % Voxel_grid_cols;
            vcamis = floor(start / Voxel_grid_cols);
            ucamie = end % Voxel_grid_cols;
            vcamie = floor(end / Voxel_grid_cols);

            if (ucamis * stepSize_rows > 6 && vcamis > 6 * stepSize_cols) {
                starto = cv::Point(vcamis*stepSize_rows, ucamis * stepSize_cols);
                endo = cv::Point(vcamie*stepSize_rows, ucamie * stepSize_cols);
                cv::line(image_trajectory, starto, endo, cv::Scalar(255, 0, 255));
            }
        }

    }
}


void Map_3D::HSVtoRGB(double &r, double &g, double &b, double h, double s, double v) {
    int i;
    double f, p, q, t;

    if (s == 0) {
        // achromatic (grey)
        r = g = b = v;
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
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default: // case 5:
            r = v;
            g = p;
            b = q;
            break;
    }

}

/**
 * Show 3D MAp 
 */
void Map_3D::show_3D_2Dmap(){
    
     //initialize the background
    for (int u = 0; u < Voxel_grid_cols * stepSize_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows * stepSize_rows; v++) {
            image_3D.at<cv::Vec3b>(u, v)[0] = 0;
            image_3D.at<cv::Vec3b>(u, v)[1] = 0;
            image_3D.at<cv::Vec3b>(u, v)[2] = 0;
        }
    }
    
    int nu;
    int nv;
    double H, R, G, B;
    
    //color in the image ground,obstercal,camera 
    for (int u = 0; u < Voxel_grid_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows; v++) {
            //for each unit$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            for (int a = 1; a < stepSize_rows - 1; a++) {
                for (int b = 1; b < stepSize_cols - 1; b++) {
                    nu = u * stepSize_cols + b;
                    nv = v * stepSize_rows + a;
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    if (nu > 0 && nv > 0 && nu < image_3D.rows && nv < image_3D.cols) { //now here must be something different,  

                        if (ground[Voxel_grid_cols * v + u].full == true) {

                            
                            H = 0 + (1.35- ground[Voxel_grid_cols * v + u].highest) / 1.35 * 180; //choose the range of H from 0 to 180,z is from 0-2
                            HSVtoRGB(R, G, B, H, 1.0, 1.0);

                            image_3D.at<cv::Vec3b>(nu, nv)[0] = B * 255;
                            image_3D.at<cv::Vec3b>(nu, nv)[1] = G * 255;
                            image_3D.at<cv::Vec3b>(nu, nv)[2] = R * 255;

                        }    
                       
                        else if (ground[Voxel_grid_cols * v + u].is_floor == true) {
                            image_3D.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_3D.at<cv::Vec3b>(nu, nv)[1] = 255;
                            image_3D.at<cv::Vec3b>(nu, nv)[2] = 255;
                            
                        }
                        
                        if(ground[Voxel_grid_cols * v + u].is_camera== true) {
                            image_3D.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_3D.at<cv::Vec3b>(nu, nv)[1] = 0;
                            image_3D.at<cv::Vec3b>(nu, nv)[2] = 255;
                                   
                        }                       
                    }
                }
            } 
        }
    }
    
    
}

void Map_3D::zoom_img(Eigen::Matrix4d T) {

    int index = compute_cami_index(T);
    int cam_u = index % Voxel_grid_cols;
    int cam_v = floor(index / Voxel_grid_rows); //here the cam_v and cam_u are in the original img,from the left conner

    //initialize the background
    for (int u = 0; u < 100 * stepSize_cols; u++) {
        for (int v = 0; v < 100 * stepSize_rows; v++) {
            image_trajectory.at<cv::Vec3b>(u, v)[0] = 0;
            image_trajectory.at<cv::Vec3b>(u, v)[1] = 0;
            image_trajectory.at<cv::Vec3b>(u, v)[2] = 0;
        }
    }

    int i = 0;
    int j = 0;
    for (int u = (cam_u - 50) * stepSize_cols; u < (cam_u + 50) * stepSize_cols; u++) {
        if (cam_u - 50 < 0) continue;
        for (int v = (cam_v - 50) * stepSize_rows; v < (cam_v + 50) * stepSize_rows; v++) {
            if (cam_v - 50 < 0) continue;
            image_trajectory.at<cv::Vec3b>(i, j)[0] = image_3D.at<cv::Vec3b>(u, v)[0];
            image_trajectory.at<cv::Vec3b>(i, j)[1] = image_3D.at<cv::Vec3b>(u, v)[1];
            image_trajectory.at<cv::Vec3b>(i, j)[2] = image_3D.at<cv::Vec3b>(u, v)[2];

            j++;
        }
        i++;
    }
}

void Map_3D::voxel_cloud(pcl::PointCloud<PointType> &cloud_in) {

    //define the struct of the pointcloud
    pcl::PointCloud<PointType>::Ptr CLOUD_IN(new pcl::PointCloud<PointType>);
    CLOUD_IN->width = Voxel_grid_rows * Voxel_grid_cols*Voxel_grid_higs;
    CLOUD_IN->height = 1;
    CLOUD_IN->points.resize(CLOUD_IN->width * CLOUD_IN->height);
    int index_point;
    double H, R, G, B;
    //Eigen::Matrix4d T_CT_CAMI;
    Eigen::Vector4d coor=T_CT_CAMI*Eigen::Vector4d(0,0,0,1);

    for (int u = 0; u < Voxel_grid_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows; v++) {

            double offsetx = -Voxel_grid_cols * unit_cols / 2;
            double offsety = -Voxel_grid_rows * unit_rows / 2;

            if (ground[Voxel_grid_cols * v + u].full == true) {

                CLOUD_IN->points[Voxel_grid_cols * v + u].x = -((u + 0.5) * unit_cols + offsetx);
                CLOUD_IN->points[Voxel_grid_cols * v + u].y = -((v + 0.5) * unit_cols + offsety);
                CLOUD_IN->points[Voxel_grid_cols * v + u].z = 0.5 * unit_height;

                if(sqrt(pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].x-coor[0],2)+pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].y-coor[1],2))<Radius){
                CLOUD_IN->points[Voxel_grid_cols * v + u].r = 255;
                CLOUD_IN->points[Voxel_grid_cols * v + u].g = 255;
                CLOUD_IN->points[Voxel_grid_cols * v + u].b = 255;
                }
                else{
                CLOUD_IN->points[Voxel_grid_cols * v + u].r = 155;
                CLOUD_IN->points[Voxel_grid_cols * v + u].g = 155;
                CLOUD_IN->points[Voxel_grid_cols * v + u].b = 155;               
                }

                for (int j = 0; j < Voxel_grid_higs; j++) {
                    index_point = Voxel_grid_rows * Voxel_grid_cols * j + Voxel_grid_cols * v + u;
                    if (ground[Voxel_grid_cols * v + u].to.voxels[j].counter >= thre_counter) { //considering the counter as fillter

                        CLOUD_IN->points[index_point].x = -((u + 0.5) * unit_cols + offsetx);
                        CLOUD_IN->points[index_point].y = -((v + 0.5) * unit_rows + offsety);
                        CLOUD_IN->points[index_point].z = (j + 0.5) * unit_height;

                        if(sqrt(pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].x-coor[0],2)+pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].y-coor[1],2))<Radius){
                        H = 0 + (1.35 - (j + 0.5) * unit_height) / 1.35 * 180; //choose the range of H from 0 to 180,z is from 0-2
                        HSVtoRGB(R, G, B, H, 1.0, 1.0);
                        CLOUD_IN->points[index_point].r = R * 255;
                        CLOUD_IN->points[index_point].g = G * 255;
                        CLOUD_IN->points[index_point].b = B * 255;
                        }
                        else {
                            CLOUD_IN->points[index_point].r = 155;
                            CLOUD_IN->points[index_point].g = 155;
                            CLOUD_IN->points[index_point].b = 155;
                        }
                    }
                }
            } else if (ground[Voxel_grid_cols * v + u].is_floor == true) {

                CLOUD_IN->points[Voxel_grid_cols * v + u].x = -((u + 0.5) * unit_cols + offsetx);
                CLOUD_IN->points[Voxel_grid_cols * v + u].y = -((v + 0.5) * unit_cols + offsety);
                CLOUD_IN->points[Voxel_grid_cols * v + u].z = 0.5 * unit_height;
                
                
                if(sqrt(pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].x-coor[0],2)+pow(-CLOUD_IN->points[Voxel_grid_cols * v + u].y-coor[1],2))<Radius){
                CLOUD_IN->points[Voxel_grid_cols * v + u].r = 255;
                CLOUD_IN->points[Voxel_grid_cols * v + u].g = 255;
                CLOUD_IN->points[Voxel_grid_cols * v + u].b = 255;
                }
                
                else{
                CLOUD_IN->points[Voxel_grid_cols * v + u].r = 155;
                CLOUD_IN->points[Voxel_grid_cols * v + u].g = 155;
                CLOUD_IN->points[Voxel_grid_cols * v + u].b = 155;
                    
                }

            }
        }
    }

    cloud_in = (*CLOUD_IN);

}