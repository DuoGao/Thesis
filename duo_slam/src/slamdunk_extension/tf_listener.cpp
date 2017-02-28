/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tf_listener.cpp
 * Author: Duo Gao
 * 
 * Created on 13 luglio 2016, 10.49
 */

#include "tf_listener.h"

tf_listener::tf_listener() {
}

tf_listener::tf_listener(const tf_listener& orig) {
}

tf_listener::~tf_listener() {
}

void tf_listener::init_T_Asus_cami(){ //if it doesnt work just try to make the matrix inverse and retry
    
   T_Asus_cami=T_Asus_cami.setZero();
   T_Asus_cami(0, 2) = 1;
   T_Asus_cami(1, 0) = -1;
   T_Asus_cami(2, 1) = 1;
   T_Asus_cami(3, 3) = 1;
   //Translation
   T_Asus_cami(0, 3) = 0.0;
   T_Asus_cami(1, 3) = -0.01;
   T_Asus_cami(2, 3) = -0.02;
   
}

void tf_listener::compute_cami_uv(Eigen::Matrix4d T,int &U,int &V) {
    
    double ucam, vcam;
    Eigen::Vector4d camera_position = Eigen::Vector4d(
            T(0, 3),
            T(1, 3),
            T(2, 3),
            1.0);
    double c_x = camera_position[0] + Voxel_grid_cols * unit_cols / 2;
    double c_y = camera_position[1] + Voxel_grid_rows * unit_rows / 2;
    
    modf(c_x / unit_cols, &ucam);
    modf(c_y / unit_rows, &vcam);
    U = (int)ucam;
    V = (int)vcam;

}


void tf_listener::cami_VO_eigen( Eigen::Matrix4d &T,Eigen::Matrix4d &T_o) {
    
    try {
        listener.lookupTransform("/vicon", "/Asus", ros::Time(0), transform);  //"/Asus", "/vicon"
        listener.lookupTransform("/odom", "/camera_rgb_optical_frame", ros::Time(0), transform_o); 
        //printf("transform.getOrigin().getX() is %f \n", transform.getOrigin().getX());
    }   
    catch (tf::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
        //ros::Duration(1.0).sleep();  
        // continue;   need to pay attention if the continue is delete
    }
    
    Utility::eigen_4x4_d_to_tf(T,transform,true);
    Utility::eigen_4x4_d_to_tf(T_o,transform_o,true);

}


void tf_listener::vicon_cami_in_1stref(Eigen::Matrix4d T_CAMFIRST_CT){ 
    //represent the cam saw by vicon in CT ref
    cami_VO_eigen(T_vicon_Asus,T_odom_cami); 
    //printf("T_vicon_cami(1,1) is %f \n", T_vicon_cami(1,1));
    T_vicon_cami=T_vicon_Asus*T_Asus_cami;
    
    T_CT_vicon=T_CAMFIRST_CT.inverse()*T_vicon_camfirst.inverse();
    T_CT_camiv = T_CT_vicon*T_vicon_cami;
    
    compute_cami_uv(T_CT_camiv, tf_camiu_vicon, tf_camiv_vicom);
       
}

void tf_listener::visualize_cami_VO(cv::Mat &ima) { 
    int nu, nv;
    int nuo, nvo;
    //for each unit$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    for (int a = 1; a < stepSize_rows - 1; a++) {
        for (int b = 1; b < stepSize_cols - 1; b++) {
            nu = tf_camiu_vicon * stepSize_cols + b;
            nv = tf_camiv_vicom * stepSize_rows + a;

            nuo = tf_camiu_odom * stepSize_cols + b;
            nvo = tf_camiv_odom * stepSize_rows + a;
            //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            if (nu > 0 && nv > 0 && nu < ima.rows && nv < ima.cols) {
                ima.at<cv::Vec3b>(nu, nv)[0] = 0;
                ima.at<cv::Vec3b>(nu, nv)[1] = 255;
                ima.at<cv::Vec3b>(nu, nv)[2] = 0;

            }
            if (nuo > 0 && nvo > 0 && nuo < ima.rows && nvo < ima.cols) {
                ima.at<cv::Vec3b>(nuo, nvo)[0] = 255;
                ima.at<cv::Vec3b>(nuo, nvo)[1] = 255;
                ima.at<cv::Vec3b>(nuo, nvo)[2] = 0;
            }
        }
    }
    
    
    if(nu>6&&nv>6&&nu_pre_vicon>6&&nv_pre_vicon>6){
        
        cv::Point startv = cv::Point(nv_pre_vicon,nu_pre_vicon);
        cv::Point endv = cv::Point(nv,nu);
        cv::line(ima,startv,endv,cv::Scalar(0,255,0));  
        
        cv::Point starto = cv::Point(nv_pre_odom,nu_pre_odom);
        cv::Point endo = cv::Point(nvo,nuo);
        cv::line(ima,starto,endo,cv::Scalar(255,255,0)); 
        
        //cv::Point starto = cv::Point(nvo_pre,nuo_pre);
        //cv::Point endo = cv::Point(nvo,nuo);
        //cv::line(ima,starto,endo,cv::Scalar(255,255,0));        
        
    }
    nu_pre_vicon=nu;
    nv_pre_vicon=nv;
    nu_pre_odom=nuo;
    nv_pre_odom=nvo;  
}


void tf_listener::odom_cami_in_1stref(Eigen::Matrix4d T_CAMFIRST_CT) {

   // cami_odom_eigen(T_odom_cami);

    T_CT_odom = T_CAMFIRST_CT.inverse() * T_odom_camfirst.inverse();
    T_CT_camio = T_CT_odom*T_odom_cami;

    compute_cami_uv(T_CT_camio, tf_camiu_odom, tf_camiv_odom);
}

