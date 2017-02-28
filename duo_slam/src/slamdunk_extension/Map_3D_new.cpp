/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Map_3D_new.cpp
 * Author: Duo Gao
 * 
 * Created on August 1, 2016, 6:05 PM
 */

#include "Map_3D_new.h"

Map_3D_new::Map_3D_new() {
}

Map_3D_new::Map_3D_new(const Map_3D_new& orig) {
}

Map_3D_new::~Map_3D_new() {
}

int Map_3D_new::compute_cami_index(Eigen::Matrix4d T){
    
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

//void Map_3D_new::update_heightest(voxel_block* head) {
//
//    voxel_block *cur = head;
//    while (cur != NULL) {
//        if (cur->next == NULL) {
//            cur->voxels
//        }
//        cur = cur->next;
//    }
//
//}

void Map_3D_new::add_or_minus_3D_voxel(Eigen::Matrix4d T,double x, double y, double z,int cloud_mode,int add_minus_mode) {    

    Eigen::Vector4d point_position = Eigen::Vector4d(x, y, z, 1.0);
    double offset_u = Voxel_grid_cols*unit_cols / 2.0;
    double offset_v = Voxel_grid_rows*unit_rows / 2.0;

    point_position = T*point_position;
    double X = point_position[0] + offset_u;
    double Y = point_position[1] + offset_v;
    double H = point_position[2];
    double U,V,J;
    modf(X / unit_cols,&U);
    modf(Y / unit_rows, &V);
    modf(H / unit_height, &J);
    
    int array_index_2D = (int) Voxel_grid_cols * V + U;     
    int voxel_seq=(int)J;                                                                   //voxel index in height is start from zero
    int cami_index;

    if (cloud_mode == mode_obs) {
        if (V < Voxel_grid_rows && U < Voxel_grid_cols && J < Voxel_grid_higs)      //here should be modified since there is no height limit
            if (V >= 0 && U >= 0 && J >= 0) {
 //-------------------------------------------------------------------------------------------------------------------------------------------------------              
                if (!ground[array_index_2D].full) { 
                                                                                //all the things here can be made into a function!!!!!!!!!!
                    int list_seq=in_which_list(voxel_seq);  
                    ground[array_index_2D].full = true;
                    ground[array_index_2D].VOXEL_BLOCKS= new VOXEL_BLOCK(list_seq,list_num);
                    int local_index=local_index(voxel_seq, ground[array_index_2D].VOXEL_BLOCKS);
                    ground[array_index_2D].VOXEL_BLOCKS->voxels[local_index].counter++;
                    ground[array_index_2D].highest=voxel_seq*unit_height;
                }

                else {                   

                    std::string operation;   
                    operation=check_linked_list(ground[array_index_2D].VOXEL_BLOCKS,voxel_seq,add_minus_mode);  //mode only works when exsiting
                    
                    //update the heighest ,when a new point comes,but here actually some problem when the highest point has been deleted
                    if(ground[array_index_2D].highest<(voxel_seq+1)*unit_height){
                       ground[array_index_2D].highest=(voxel_seq+1)*unit_height;     
                    }

                    //update the camera position according the mode,when something change also the position of the camera will change
                    cami_index = compute_cami_index(T);
                    if (add_minus_mode == integrate_mode)
                        ground[cami_index].is_camera = true;
                    else
                        ground[cami_index].is_camera = false;
                    } 
//---------------------------------------------------------------------------------------------------------------------------------------------------------           
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
void Map_3D_new::compute_3D_voxel_cloud(Eigen::Matrix4d T, pcl::PointCloud<PointType>& cloud_obs, int mode) { 

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


void Map_3D_new::integrate_clouds(Eigen::Isometry3d pose, pcl::PointCloud<PointType>& cloud_obs, pcl::PointCloud<PointType>& cloud_ground, int mode){
    
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

        add_or_minus_3D_voxel(T_CT_CAMI, cloud_GROUND->points[i].x, cloud_GROUND->points[i].y, cloud_GROUND->points[i].z, mode_plane,mode); 

    }
      
    
}


void Map_3D_new::update_trajectory(Eigen::Isometry3d pose,std::string key){
        
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

void Map_3D_new::trajectory_slamdunk_cami() {

    int start, end, ucamis, vcamis, ucamie, vcamie;
    cv::Point starto;
    cv::Point endo;
    for (int u = 0; u < Voxel_grid_cols * stepSize_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows * stepSize_rows; v++) {
            if (image_trajectory_new.at<cv::Vec3b>(u, v)[0] == 255 && image_trajectory_new.at<cv::Vec3b>(u, v)[1] == 0 && image_trajectory_new.at<cv::Vec3b>(u, v)[2] == 255) {
                image_trajectory_new.at<cv::Vec3b>(u, v)[0] = 255;
                image_trajectory_new.at<cv::Vec3b>(u, v)[1] = 255;
                image_trajectory_new.at<cv::Vec3b>(u, v)[2] = 255;
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
                cv::line(image_trajectory_new, starto, endo, cv::Scalar(255, 0, 255));
            }
        }

    }
}


void Map_3D_new::HSVtoRGB(double &r, double &g, double &b, double h, double s, double v) {
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
void Map_3D_new::show_3D_2Dmap(){
    
     //initialize the background
    for (int u = 0; u < Voxel_grid_cols * stepSize_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows * stepSize_rows; v++) {
            image_3D_new.at<cv::Vec3b>(u, v)[0] = 0;
            image_3D_new.at<cv::Vec3b>(u, v)[1] = 0;
            image_3D_new.at<cv::Vec3b>(u, v)[2] = 0;
        }
    }
    
    int nu;
    int nv;
    //color in the image ground,obstercal,camera 
    for (int u = 0; u < Voxel_grid_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows; v++) {
            //for each unit$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            for (int a = 1; a < stepSize_rows - 1; a++) {
                for (int b = 1; b < stepSize_cols - 1; b++) {
                    nu = u * stepSize_cols + b;
                    nv = v * stepSize_rows + a;
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    if (nu > 0 && nv > 0 && nu < image_3D_new.rows && nv < image_3D_new.cols) { //now here must be something different,  

                        if (ground[Voxel_grid_cols * v + u].full == true) {

                            double H, R, G, B;
                            H = 0 + (2 - ground[Voxel_grid_cols * v + u].highest) / 2 * 180; //choose the range of H from 0 to 180,z is from 0-2
                            HSVtoRGB(R, G, B, H, 1.0, 1.0);

                            image_3D_new.at<cv::Vec3b>(nu, nv)[0] = B * 255;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[1] = G * 255;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[2] = R * 255;

                        }    
                       
                        else if (ground[Voxel_grid_cols * v + u].is_floor == true) {
                            image_3D_new.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[1] = 255;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[2] = 255;
                            
                        }
                        
                        if(ground[Voxel_grid_cols * v + u].is_camera== true) {
                            image_3D_new.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[1] = 0;
                            image_3D_new.at<cv::Vec3b>(nu, nv)[2] = 255;
                                   
                        }                       
                    }
                }
            } 
        }
    }
    
    
}



void Map_3D_new::voxel_cloud(pcl::PointCloud<PointType> &cloud_in) {
    
    double offsetx,offsety;
    int ground_index;

    //define the struct of the pointcloud
    pcl::PointCloud<PointType>::Ptr CLOUD_IN(new pcl::PointCloud<PointType>);
    CLOUD_IN->width = Voxel_grid_rows * Voxel_grid_cols*Voxel_grid_higs;
    CLOUD_IN->height = 1;
    CLOUD_IN->points.resize(CLOUD_IN->width * CLOUD_IN->height);

    for (int u = 0; u < Voxel_grid_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows; v++) {

            offsetx = -Voxel_grid_cols * unit_cols / 2;
            offsety = -Voxel_grid_rows * unit_rows / 2;
            ground_index=Voxel_grid_cols * v + u;
            
            if (ground[ground_index].full == true) {
                
                //make sure there is floor under the obs
                CLOUD_IN->points[ground_index].x = -((u + 0.5) * unit_cols + offsetx);
                CLOUD_IN->points[ground_index].y = -((v + 0.5) * unit_cols + offsety);
                CLOUD_IN->points[ground_index].z = 0.5 * unit_height;
                CLOUD_IN->points[ground_index].r = 255;
                CLOUD_IN->points[ground_index].g = 255;
                CLOUD_IN->points[ground_index].b = 255;
//--------------------------------------------------------------------------------------------------------------------------------------------
                //assign each voxel into the right position
                voxel_block *cur = ground[ground_index].VOXEL_BLOCKS;
                while (cur!= NULL) {
                    for(int i=0;i<list_num;i++){    
                        
                    int j=(int)i+cur->index*list_num;   
                    int index_point = Voxel_grid_rows * Voxel_grid_cols * j + Voxel_grid_cols * v + u;       
                    
                    if (cur->voxels[i].counter>= thre_counter) {

                        CLOUD_IN->points[index_point].x = -((u + 0.5) * unit_cols + offsetx);
                        CLOUD_IN->points[index_point].y = -((v + 0.5) * unit_rows + offsety);
                        CLOUD_IN->points[index_point].z = (j + 0.5) * unit_height;

                        double H, R, G, B;
                        H = 0 + (2 - (j + 0.5) * unit_height) / 2 * 180;
                        HSVtoRGB(R, G, B, H, 1.0, 1.0);

                        CLOUD_IN->points[index_point].r = R * 255;
                        CLOUD_IN->points[index_point].g = G * 255;
                        CLOUD_IN->points[index_point].b = B * 255;

                    }
                    
                    }
                    cur = cur->next;
                }
                
//----------------------------------------------------------------------------------------------------------------------------------------------                
            }
            else if (ground[Voxel_grid_cols * v + u].is_floor == true) {
                int index_pointf = Voxel_grid_cols * v + u;
                CLOUD_IN->points[index_pointf].x = -((u + 0.5) * unit_cols + offsetx);
                CLOUD_IN->points[index_pointf].y = -((v + 0.5) * unit_cols + offsety);
                CLOUD_IN->points[index_pointf].z = 0.5 * unit_height;

                CLOUD_IN->points[index_pointf].r = 255;
                CLOUD_IN->points[index_pointf].g = 255;
                CLOUD_IN->points[index_pointf].b = 255;
            }
        }
    }

    cloud_in = (*CLOUD_IN);

}

//for linked list
/////////////////
/////////////////
//add voxel_block and also add the counter,not consider the add and minus mode 
void Map_3D_new::addNode(voxel_block* head,int voxel_seq,int mode) {
    int list_seq=in_which_list(voxel_seq);
    voxel_block *newvoxel = new VOXEL_BLOCK(list_seq,list_num);
    voxel_block *cur = head;
    int local_index=local_index(voxel_seq,cur);
    
    while (cur!=NULL) {
        if (cur->next == NULL) {
            cur->next = newvoxel;
            
            cur->next->voxels[local_index].counter++;
            
            return;
        }
        cur = cur->next;
    }

}

//this time the pointer head is one node ahead
void Map_3D_new::insertFront(voxel_block** head,int voxel_seq) {
    int list_seq = in_which_list(voxel_seq);
    voxel_block *newvoxel = new VOXEL_BLOCK(list_seq, list_num);
    newvoxel->next = *head;
    *head = newvoxel;
}

//j is the voxel number,notice the number of the big block is from zero
int Map_3D_new::in_which_list(int j) {
    int num_list = (int)j / list_num;
    return num_list;
}

int Map_3D_new::local_index(int voxel_seq,voxel_block* cur){
    float voxel_height=voxel_seq*unit_height;
    int local_index=(int)(voxel_height-cur->bottomSurfce())/unit_height;
    return local_index;
}

//put the local index totally outside,here we only fixed the position
std::string Map_3D_new::check_linked_list(voxel_block* head,int voxel_seq,int mode){
    float topest_cur;
    std::string name;
    float voxel_height=voxel_seq*unit_height;
    voxel_block* cur=head;      
      
    while(cur!=NULL){
               
        if (cur->bottomSurfce()<=voxel_height&&cur->topSurfce()>voxel_height){                 //situation1: point in the existing voxel_block 
            name="existing";      
            int local_index=local_index(voxel_seq,cur);
            
            if (mode == integrate_mode)
                cur->voxels[local_index].counter++;
            if (mode == deintegrate_mode)
                cur->voxels[local_index].counter-=1;
            return name;
        }
        if (cur->topSurfce() <= voxel_height && cur->next->bottomSurfce() > voxel_height) {    //situation2: insert the node and add the point
            name = "insert";
            insertFront(& cur->next,voxel_seq);
            int local_index_i = local_index(voxel_seq, cur->next);
            cur->next->voxels[local_index_i].counter++;
            return name;
        }
        topest_cur=cur->topSurfce();
        cur = cur->next;
    }

    if (topest_cur = <voxel_height) {                                                          //situation3: append to the head
        name = "append";
        addNode(cur,voxel_seq);    
        return name;
    } 

}