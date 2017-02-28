/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tf_listener.h
 * Author: Duo Gao
 *
 * Created on 13 luglio 2016, 10.49
 */

#ifndef TF_LISTENER_H
#define TF_LISTENER_H

#include "Utility.h"

class tf_listener {
public:

    tf_listener();
    tf_listener(const tf_listener& orig);
    virtual ~tf_listener();
    
    //things deal with the vicon
    tf::TransformListener listener;
    tf::StampedTransform transform;
   
    Eigen::Matrix4d T_Asus_cami;
    Eigen::Matrix4d T_vicon_Asusfirst;
    Eigen::Matrix4d T_vicon_camfirst;
    
    Eigen::Matrix4d T_vicon_Asus;
    Eigen::Matrix4d T_vicon_cami;
    
    Eigen::Matrix4d T_CT_vicon;
    Eigen::Matrix4d T_CT_camiv;
    
    int tf_camiu_vicon, tf_camiv_vicom,nu_pre_vicon,nv_pre_vicon;
    
    //things deal with the odom
    tf::StampedTransform transform_o;
    
    Eigen::Matrix4d T_odom_camfirst;
    Eigen::Matrix4d T_odom_cami;
    
    Eigen::Matrix4d T_CT_odom;
    Eigen::Matrix4d T_CT_camio;
    
    int tf_camiu_odom, tf_camiv_odom,nu_pre_odom,nv_pre_odom;
     
    //things deal with the vicon
    void init_T_Asus_cami();
    void cami_VO_eigen(Eigen::Matrix4d &T,Eigen::Matrix4d &T_o);
    void vicon_cami_in_1stref(Eigen::Matrix4d T_CAMFIRST_CT);    
    void odom_cami_in_1stref(Eigen::Matrix4d T_CAMFIRST_CT);    
    void visualize_cami_VO(cv::Mat &ima);     
      
private:
    
    void compute_cami_uv(Eigen::Matrix4d T_CT_camiv,int &U,int &V);

};

#endif /* TF_LISTENER_H */

