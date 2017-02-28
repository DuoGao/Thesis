/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Utility.cpp
 * Author: Duo Gao
 * 
 * Created on 13 luglio 2016, 15.35
 */

#include "Utility.h"

Utility::Utility() {
}

Utility::Utility(const Utility& orig) {
}

Utility::~Utility() {
}

/**
 * KDL Frame to Eigen Matrix 4x4
 * @param frame KDL Frame
 * @param mat Eigen Matrix 4x4
 */
void Utility::kdl_to_eigen_4x4_d(KDL::Frame& frame, Eigen::Matrix4d& mat) {
        for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                        mat(i, j) = frame.M(i, j);
                }
        }

        mat(0, 3) = frame.p[0];
        mat(1, 3) = frame.p[1];
        mat(2, 3) = frame.p[2];
        mat(3, 3) = 1;
}

/**
 * Creates KDL Frame
 * @param x
 * @param y
 * @param z
 * @param roll
 * @param pitch
 * @param yaw
 * @param out_frame OUTPUT
 */
void Utility::create_kdl_frame(float x, float y, float z, float roll, float pitch, float yaw, KDL::Frame& out_frame) {

        out_frame.M = KDL::Rotation::RPY(
                roll,
                pitch,
                yaw
                );

        out_frame.p[0] = x;
        out_frame.p[1] = y;
        out_frame.p[2] = z;
}

/**
 * Creates Eigen4x4 Matrix
 * @param x
 * @param y
 * @param z
 * @param roll
 * @param pitch
 * @param yaw
 * @param mat
 */

void Utility::create_eigen_4x4_d(float x, float y, float z, float roll, float pitch, float yaw, Eigen::Matrix4d& mat) {
        mat = Eigen::Matrix4d::Identity();
        KDL::Frame frame;
        create_kdl_frame(x, y, z, roll, pitch, yaw, frame);
        kdl_to_eigen_4x4_d(frame, mat);
}

/**
 * Builds TF from geometry_msgs::Pose TODO: reverse
 */

void Utility::eigen_4x4_to_geometrypose_d(Eigen::Matrix4d& mat,geometry_msgs::Pose& pose){

        pose.position.x = mat(0,3);
        pose.position.y = mat(1,3);
        pose.position.z = mat(2,3);

        tf::Matrix3x3 m;
        for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++)
                        m[i][j]=mat(i,j);

        tf::Quaternion q;
        m.getRotation(q);
        q.normalize();
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
}

/**
 * Builds TF from geometry_msgs::Pose TODO: reverse
 */
void Utility::eigen_4x4_d_to_tf(Eigen::Matrix4d& t,  tf::Transform& tf, bool reverse = false){
        if(!reverse) {
                tf.setOrigin( tf::Vector3(
                                      t(0,3),
                                      t(1,3),
                                      t(2,3)
                                      ));

                tf::Matrix3x3 rot;
                for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                                rot[i][j] = t(i,j);

                tf::Quaternion q;
                rot.getRotation(q);
                q.normalize();
                tf.setRotation(q);
        }else{
                t = Eigen::Matrix4d::Identity();
                t(0,3) = tf.getOrigin()[0];
                t(1,3) = tf.getOrigin()[1];
                t(2,3) = tf.getOrigin()[2];

                tf::Matrix3x3 rot;
                rot.setRotation(tf.getRotation());
                for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                                t(i,j) = rot[i][j];

        }
}

void Utility::draw_reference_frame(pcl::visualization::PCLVisualizer &viewer,  Eigen::Matrix4d& rf, float size, std::string name){


        Eigen::Vector3f center;
        center << rf(0,3),rf(1,3),rf(2,3);

        Eigen::Vector3f ax;
        Eigen::Vector3f ay;
        Eigen::Vector3f az;

        ax << rf(0,0),rf(1,0),rf(2,0);
        ay << rf(0,1),rf(1,1),rf(2,1);
        az << rf(0,2),rf(1,2),rf(2,2);

        ax = ax * size + center;
        ay = ay * size + center;
        az = az * size + center;

        std::stringstream ss;

        ss << name << "_x";
        draw_3D_vector(viewer, center, ax, 1, 0, 0, ss.str().c_str());
        ss << name << "_y";
        draw_3D_vector(viewer, center, ay, 0, 1, 0, ss.str().c_str());
        ss << name << "_z";
        draw_3D_vector(viewer, center, az, 0, 0, 1, ss.str().c_str());

}

void  Utility::convert_point_3D(PointType& pt, Eigen::Vector3f& p, bool reverse) {
    if (!reverse) {
        p[0] = pt.x;
        p[1] = pt.y;
        p[2] = pt.z;
    } else {
        pt.x = p[0];
        pt.y = p[1];
        pt.z = p[2];
    }
}

void Utility::draw_3D_vector(pcl::visualization::PCLVisualizer& viewer, Eigen::Vector3f start, Eigen::Vector3f end, float r, float g, float b, std::string name) {
    PointType p_start, p_end;
    convert_point_3D(p_start, start, true);

    Eigen::Vector3f dv;
    dv = end - start;

    p_end.x = p_start.x + dv[0];
    p_end.y = p_start.y + dv[1];
    p_end.z = p_start.z + dv[2];

    viewer.addArrow(p_end, p_start, r, g, b, false, name);
}