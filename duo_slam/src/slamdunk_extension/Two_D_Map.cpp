/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Two_D_Map.cpp
 * Author: Duo Gao
 * 
 * Created on 24 giugno 2016, 10.51
 */

#include "Two_D_Map.h"

Two_D_Map::Two_D_Map() {
}

Two_D_Map::Two_D_Map(const Two_D_Map& orig) {
}

Two_D_Map::~Two_D_Map() {
}

void Two_D_Map::init_Voxel_grid() {

    for (int i = 0; i < Voxel_grid_rows*Voxel_grid_cols; ++i) {
        Voxel_grid[i].height = 0.0;
        Voxel_grid[i].occupied = false;
        Voxel_grid_ground[i].height = 0.0;
        Voxel_grid_ground[i].occupied = false;  
        Voxel_grid_camera[i].height = 0.0;
        Voxel_grid_camera[i].occupied = false;  
    }
}

double Two_D_Map::Round_planez(double Height, int precision) {
    double result;
    result = round(Height * precision) / precision;
    return result;
}


void Two_D_Map::map_cloud_fillter(pcl::PointCloud<PointType> &input_cloud, double voxel_leaf) {
    pcl::PointCloud<PointType>::Ptr INPUT(new pcl::PointCloud<PointType>);
    (*INPUT) = input_cloud;
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(INPUT);
    sor.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
    sor.filter(*INPUT);
    input_cloud = (*INPUT);
}

void Two_D_Map::RadiusOutlierRemoval(pcl::PointCloud<PointType> &input_cloud, double radius, double min_neighborhood) {
    pcl::PointCloud<PointType>::Ptr INPUT(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr OUTPUT(new pcl::PointCloud<PointType>);
    (*INPUT) = input_cloud;
    
    //the condition filter
    // build the condition
    pcl::ConditionAnd<PointType>::Ptr range_cond (new pcl::ConditionAnd<PointType> ());
    //range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, 0.1)));
    range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::GT, -0.2)));
    range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::LT, 0.2)));
    // build the filter
    pcl::ConditionalRemoval<PointType> condrem;
    condrem.setCondition (range_cond);
    condrem.setInputCloud (INPUT);
    condrem.setKeepOrganized(true);
    // apply filter
    condrem.filter (*OUTPUT);
    
    
    
//    pcl::RadiusOutlierRemoval<PointType> outrem;
//    outrem.setInputCloud(INPUT);
//    outrem.setRadiusSearch(radius);
//    outrem.setMinNeighborsInRadius(min_neighborhood);
//    outrem.filter(*OUTPUT);
    input_cloud = (*OUTPUT);
}

void Two_D_Map::pass_through_fillter(pcl::PointCloud<PointType> &input_cloud, double max, double min, bool setnagative) {
    pcl::PointCloud<PointType>::Ptr INPUT(new pcl::PointCloud<PointType>);
    (*INPUT) = input_cloud;
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud(INPUT);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min, max);
    pass.setFilterLimitsNegative(setnagative);
    pass.filter(*INPUT);
    input_cloud = (*INPUT);
}

void Two_D_Map::project_1stnormal_ref(Cloudinfo &plane_cloud, Eigen::Matrix4d &T_CAMFIRST_CT1) {

    Eigen::Vector3d orth_x;
    Eigen::Vector3d orth_y;
    Eigen::Vector3d first_plane_normal;
    Eigen::Vector4d centroid;

    pcl::compute3DCentroid(*(plane_cloud.input_cloud), plane_cloud.plane_indices.indices, centroid);
    first_plane_normal = plane_cloud.plane_normal;

    orth_x = Eigen::Vector3d(1.0, 1.0, (-first_plane_normal[0] - first_plane_normal[1]) / first_plane_normal[2]);
    orth_x.normalize();
    orth_y = first_plane_normal.cross(orth_x);
    orth_y.normalize();

    T_CAMFIRST_CT1 = Eigen::Matrix4d::Identity();
    T_CAMFIRST_CT1(0, 2) = first_plane_normal[0];
    T_CAMFIRST_CT1(1, 2) = first_plane_normal[1];
    T_CAMFIRST_CT1(2, 2) = first_plane_normal[2];

    T_CAMFIRST_CT1(0, 3) = centroid[0];
    T_CAMFIRST_CT1(1, 3) = centroid[1];
    T_CAMFIRST_CT1(2, 3) = centroid[2];

    T_CAMFIRST_CT1(0, 0) = orth_x[0];
    T_CAMFIRST_CT1(1, 0) = orth_x[1];
    T_CAMFIRST_CT1(2, 0) = orth_x[2];

    T_CAMFIRST_CT1(0, 1) = orth_y[0];
    T_CAMFIRST_CT1(1, 1) = orth_y[1];
    T_CAMFIRST_CT1(2, 1) = orth_y[2];


}


//extract all the information of cloud struct

bool Two_D_Map::segment_Plane(pcl::PointCloud<PointType>::Ptr& cloud, Cloudinfo& cloud_struct, int min_inliers, int max_iterations, float distance_th, bool optimize_coefficient) {

    //Create the SimplePLane object
    cloud_struct = Cloudinfo(cloud);

    //Segmentation 
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients(optimize_coefficient);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_th);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, cloud_struct.plane_coefficients);

    if (inliers->indices.size() == 0) {
        printf("Could not estimate a planar model for the given dataset.");
        return false;
    }

    pcl::PointIndices p_indices;
    pcl::PointIndices r_indices;
    pcl::PointCloud<PointType>::Ptr p_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr r_cloud(new pcl::PointCloud<PointType>);
    // Extract the inliers of PLANES and for REST
    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(p_indices.indices);
    extract.filter(*p_cloud);
    extract.setNegative(true);
    extract.filter(r_indices.indices);
    extract.filter(*r_cloud);

    cloud_struct.plane_indices = p_indices;
    cloud_struct.rest_indices = r_indices;
    cloud_struct.plane_cloud = p_cloud;
    cloud_struct.rest_cloud = r_cloud;

    //Build the normal
    cloud_struct.build_normal(min_inliers);

    return true;
}

void Two_D_Map::first_callback_init(pcl::PointCloud<PointType>::Ptr &cloud) {
    Cloudinfo cloud1st;
    segment_Plane(cloud, cloud1st, 10, 100, 0.1, true);
    FIRST_PLANE_NORMAL = cloud1st.plane_normal;
    FIRST_PLANE_COEFFICIENTS = cloud1st.plane_coefficients;
    project_1stnormal_ref(cloud1st, T_CAMFIRST_CT);

    CLOUD_NAME.str("");
    CLOUD_NAME << "height:" << Round_planez(0.0, 10);
    FIRST_PLANE_NAME = CLOUD_NAME.str();
    MAP_PLANE[FIRST_PLANE_NAME] = (*cloud1st.plane_cloud);
    MAP_REST[FIRST_PLANE_NAME] = (*cloud1st.rest_cloud);
}



//key is with the same name of the orgin cloud

void Two_D_Map::ground_finder(pcl::PointCloud<PointType>::Ptr& cloud_in, std::string key, Eigen::Matrix4d T_S_CAM, double Angle_limit, double Height_limit, double Area_limit) {

    pcl::PointCloud<PointType>::Ptr cloud_IN(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_p(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_po(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_r(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_ro(new pcl::PointCloud<PointType>);

    T_CAMFIRST_CAMI = T_S_CAMFIRST.inverse() * T_S_CAM;
    T_CT_CAMI = T_CAMFIRST_CT.inverse() * T_CAMFIRST_CAMI;


    pcl::transformPointCloud(*cloud_in, *cloud_IN, T_CT_CAMI);

    for (size_t i = 0; i < cloud_IN->points.size(); ++i) {
        if (cloud_IN->points[i].z <= Height_limit) (*cloud_p).push_back(cloud_IN->points[i]);
        else (*cloud_r).push_back(cloud_IN->points[i]);
    }
    Eigen::Matrix4d inv = T_CT_CAMI.inverse();
    pcl::transformPointCloud(*cloud_p, *cloud_po, inv);
    pcl::transformPointCloud(*cloud_r, *cloud_ro, inv);

    //still need to seg for better result
    Cloudinfo cloud_else;
    bool seg = segment_Plane(cloud_po, cloud_else, 10, 100, 0.05, true);

    if (seg == true) {
        //Angle between prev and current
        Eigen::Vector3d current_normal = cloud_else.plane_normal;
        double angle = fabs(acos(FIRST_PLANE_NORMAL.dot(current_normal)));

        if (angle <= Angle_limit) {
            MAP_PLANE[key] = *(cloud_else.plane_cloud);
            MAP_REST[key] = (*cloud_ro)+(*(cloud_else.rest_cloud));
        } else MAP_REST[key] = (*cloud_ro);
    }
    else
        MAP_REST[key] = (*cloud_ro);

}

void Two_D_Map::project_to_1stplane(pcl::PointCloud<PointType> &cloud_obs) {
    pcl::PointCloud<PointType>::Ptr cloud_project(new pcl::PointCloud<PointType>);
    (*cloud_project) = cloud_obs;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = FIRST_PLANE_COEFFICIENTS.values[0];
    coefficients->values[1] = FIRST_PLANE_COEFFICIENTS.values[1];
    coefficients->values[2] = FIRST_PLANE_COEFFICIENTS.values[2];
    coefficients->values[3] = FIRST_PLANE_COEFFICIENTS.values[3];

    pcl::ProjectInliers<PointType> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud_project);
    proj.setModelCoefficients(coefficients);
    proj.filter(cloud_obs);

}

void Two_D_Map::compute_Voxel_grid(pcl::PointCloud<PointType> &cloud_obs,int mode) {
    int array_index;
    double X, Y, Z, U, V;
    Eigen::Vector4d point_position;
    Eigen::Vector4d camera_position;
    pcl::PointCloud<PointType>::Ptr cloud_OBS(new pcl::PointCloud<PointType>);
    (*cloud_OBS) = cloud_obs;

    //columns = sqrt(array_size);
    double side_rows = Voxel_grid_rows*unit_rows;
    double side_cols = Voxel_grid_cols*unit_cols;
    double offset_u = side_cols / 2.0;
    double offset_v = side_rows / 2.0;
    double ucam,vcam,c_x,c_y,c_z;

    for (size_t i = 0; i < cloud_OBS->size(); ++i) {

        printf("cloud_obs->points[i].z %f \n", cloud_OBS->points[i].z);
        
        //compute the index of the camera(this part should be made into a function outside )
        camera_position=Eigen::Vector4d(
                T_CT_CAMI(0,3),
                T_CT_CAMI(1,3),
                T_CT_CAMI(2,3),
                1.0);
        c_x=camera_position[0]+ offset_u;
        c_y=camera_position[1]+ offset_v;
        c_z=camera_position[2]; 
        modf(c_x / unit_cols, &ucam);
        modf(c_y / unit_rows, &vcam);
        
        int array_index_cami = (int) Voxel_grid_cols * vcam + ucam;
         
        //compute the index of the cloud point
        point_position = Eigen::Vector4d(
                cloud_OBS->points[i].x,
                cloud_OBS->points[i].y,
                cloud_OBS->points[i].z,
                1.0);

        //represent it in the first plane reference frame
        point_position = T_CT_CAMI*point_position;
        X = point_position[0] + offset_u;
        Y = point_position[1] + offset_v;
        Z = point_position[2];
        modf(X / unit_cols, &U);
        modf(Y / unit_rows, &V);
        array_index = (int) Voxel_grid_cols * V + U;
        
 
        if(mode==mode_obs){
        if (V < Voxel_grid_rows && U < Voxel_grid_cols)
            if (V >= 0 && U >= 0) 
            {   if(Z >Voxel_grid[array_index].height)        //only the height is higher than previous can be access ,plus 0.5 just for test
                Voxel_grid[array_index].height = Z;
                Voxel_grid[array_index].occupied = true;
                
                if(c_z>Voxel_grid_camera[array_index].height)
                Voxel_grid_camera[array_index_cami].height = c_z;
                Voxel_grid_camera[array_index_cami].occupied = true;
                
            }
        }
        
        if(mode==mode_plane){
        if (V < Voxel_grid_rows && U < Voxel_grid_cols)
            if (V >= 0 && U >= 0) 
            {   if(Z>Voxel_grid[array_index].height)        //only the height is higher than previous can be access ,plus 0.5 just for test
                Voxel_grid_ground[array_index].height = Z ;
                Voxel_grid_ground[array_index].occupied = true;
                
                if(c_z>Voxel_grid_camera[array_index].height)
                Voxel_grid_camera[array_index_cami].height = c_z;
                Voxel_grid_camera[array_index_cami].occupied = true;
            }
        }   
    }


}
void Two_D_Map::show_2D_map() {
    //initialize the color of the background
    for (int u = 0; u < Voxel_grid_cols * stepSize_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows * stepSize_rows; v++) {
            image_2D.at<cv::Vec3b>(u, v)[0] = 0;
            image_2D.at<cv::Vec3b>(u, v)[1] = 0;
            image_2D.at<cv::Vec3b>(u, v)[2] = 0;
        }
    }
    
    double H, R, G, B;
    //color in the image ground,obstercal,camera 
    for (int u = 0; u < Voxel_grid_cols; u++) {
        for (int v = 0; v < Voxel_grid_rows; v++) {
            //for each unit$$$$$$$$
            for (int a = 1; a < stepSize_rows - 1; a++) {
                for (int b = 1; b < stepSize_cols - 1; b++) {
                    double nu = u * stepSize_cols + b;
                    double nv = v * stepSize_rows + a;
                    //$$$$$$$$$$$$$$$$$$$$$
                    if (nu > 0 && nv > 0 && nu < image_2D.rows && nv < image_2D.cols) {

                        if (Voxel_grid[Voxel_grid_cols * v + u].occupied == true) {

                            
                            H = 0 + (Voxel_grid[Voxel_grid_cols * v + u].height) / 2 * 180; //choose the range of H from 0 to 180,z is from 0-2
                            HSVtoRGB(R, G, B, H, 1.0, 1.0);

                            image_2D.at<cv::Vec3b>(nu, nv)[0] = B * 255;
                            image_2D.at<cv::Vec3b>(nu, nv)[1] = G * 255;
                            image_2D.at<cv::Vec3b>(nu, nv)[2] = R * 255;


                        } else if (Voxel_grid_ground[Voxel_grid_cols * v + u].occupied == true) {
                            image_2D.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_2D.at<cv::Vec3b>(nu, nv)[1] = 255;
                            image_2D.at<cv::Vec3b>(nu, nv)[2] = 255;
                        } else {
                            image_2D.at<cv::Vec3b>(nu, nv)[0] = 0;
                            image_2D.at<cv::Vec3b>(nu, nv)[1] = 0;
                            image_2D.at<cv::Vec3b>(nu, nv)[2] = 0;
                        }

                        if (Voxel_grid_camera[Voxel_grid_cols * v + u].occupied == true) {
                            image_2D.at<cv::Vec3b>(nu, nv)[0] = 255;
                            image_2D.at<cv::Vec3b>(nu, nv)[1] = 0;
                            image_2D.at<cv::Vec3b>(nu, nv)[2] = 255;
                        }
                    }
                }
            }

        }
    }


}

double Two_D_Map::max_of_three(double a, double b, double c) {
    if (a >= b && a >= c) {
        return a;
    }
    if (b >= a && b >= c) {
        return b;
    }
    if (c >= a && c >= b) {
        return c;
    }

}

double Two_D_Map::min_of_three(double a, double b, double c) {
    if (a <= b && a <= c) {
        return a;
    }
    if (b <= a && b <= c) {
        return b;
    }
    if (c <= a && c <= b) {
        return c;
    }

}

void Two_D_Map::RGBtoHSV(double r, double g, double b, double &h, double &s, double &v) {
    double min, max, delta;

    min = min_of_three(r, g, b);
    max = max_of_three(r, g, b);
    v = max; // v

    delta = max - min;

    if (max != 0)
        s = delta / max; // s
    else {
        // r = g = b = 0		// s = 0, v is undefined
        s = 0;
        h = -1;
        return;
    }

    if (r == max)
        h = (g - b) / delta; // between yellow & magenta
    else if (g == max)
        h = 2 + (b - r) / delta; // between cyan & yellow
    else
        h = 4 + (r - g) / delta; // between magenta & cyan

    h *= 60; // degrees
    if (h < 0)
        h += 360;

}

void Two_D_Map::HSVtoRGB(double &r, double &g, double &b, double h, double s, double v) {
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


void Two_D_Map::fun_color(double height,double max_height,Eigen::Vector3d &bool_vector)
{ 
    double step_size;
    step_size=max_height/10;
    if(height>=0&&height<=step_size) bool_vector=Eigen::Vector3d(1,0,0);
    if(height>step_size&&height<=2*step_size) bool_vector=Eigen::Vector3d(0.5,0,0);
    if(height>2*step_size&&height<=3*step_size) bool_vector=Eigen::Vector3d(1,1,0);
    if(height>3*step_size&&height<=4*step_size) bool_vector=Eigen::Vector3d(0.5,0.5,0);
    if(height>4*step_size&&height<=5*step_size) bool_vector=Eigen::Vector3d(0,1,0);
    if(height>5*step_size&&height<=6*step_size) bool_vector=Eigen::Vector3d(0,0.5,0);
    if(height>6*step_size&&height<=7*step_size) bool_vector=Eigen::Vector3d(0,1,1);
    if(height>7*step_size&&height<=8*step_size) bool_vector=Eigen::Vector3d(0,0.5,0.5);
    if(height>8*step_size&&height<=9*max_height) bool_vector=Eigen::Vector3d(0,0,1);
    if(height>9*step_size&&height<=10*max_height) bool_vector=Eigen::Vector3d(0,0,0.5);
   
}

void Two_D_Map::create_unit_cloud(pcl::PointCloud<PointType>::Ptr &cloud_unit, double unit_size, int point_num_oneedge)//just for fun 
{
    //draw a plane point cloud the same with unit_size 
    cloud_unit->width = point_num_oneedge*point_num_oneedge;
    cloud_unit->height = 1;
    cloud_unit->points.resize(cloud_unit->width * cloud_unit->height);
    for (int i = 0; i < point_num_oneedge; ++i) {
        for (int j = 0; j < point_num_oneedge; ++j) {
            cloud_unit->points[i].x = i * unit_size / point_num_oneedge;
            cloud_unit->points[i].y = j * unit_size / point_num_oneedge;
            cloud_unit->points[i].z = 0.0;
        }
    }

}

void Two_D_Map::project_unitcloud_pcl(pcl::PointCloud<PointType>::Ptr& cloud_unit, double unit_size, double real_U, double real_V, int switch_case_phase) //just for fun
{
    //target_position is
    double t_pose_x = real_U*unit_size;
    double t_pose_y = real_V*unit_size;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 2) = t_pose_x;
    T(1, 2) = t_pose_y;

    //use the rotation rather than translation is really a smart way
    switch (switch_case_phase) {
        case 1: T(0, 0) = 1;
            T(1, 1) = 1;
            T(2, 2) = 1;
            T(3, 3) = 1;
            break; //the first phase  
        case 2: T(0, 0) = 0;
            T(0, 1) = -1;
            T(1, 0) = 1;
            T(1, 1) = 0;
            break; //the second phase
        case 3: T(0, 0) = -1;
            T(0, 1) = 0;
            T(1, 0) = 0;
            T(1, 1) = -1;
            break; //the third phase
        case 4: T(0, 0) = 0;
            T(0, 1) = 1;
            T(1, 0) = -1;
            T(1, 1) = 0;
            break; //the forth phase
        default:T(0, 1) = 1;
            T(1, 1) = 1;
    }

    //project the unit_cloud into the right positon
    pcl::transformPointCloud(*cloud_unit, *cloud_unit, T);
    pcl::transformPointCloud(*cloud_unit, *cloud_unit, T_CT_CAMI);


}
