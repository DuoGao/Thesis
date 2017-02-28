/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tsdf_cuda.h
 * Author: daniele
 *
 * Created on July 15, 2016, 12:22 PM
 */

#ifndef TSDF_CUDA_H
#define TSDF_CUDA_H

#define DEFAULT_BLOCK_DIMENSION 128


namespace slamdunk_cuda {

    typedef float SDF_VALUE_EXTENDED;
    typedef float WEIGHT_VALUE_EXTENDED;
    typedef float DATA_VALUE_EXTENDED;
    typedef short SDF_VALUE_PACKED;
    typedef short WEIGHT_VALUE_PACKED;
    typedef short DATA_VALUE_PACKED;

    typedef struct Voxel {
        SDF_VALUE_PACKED sdf;
        WEIGHT_VALUE_PACKED weight;
        DATA_VALUE_PACKED data;
    } Voxel;

    typedef struct VoxelExtended {
        SDF_VALUE_EXTENDED sdf;
        WEIGHT_VALUE_EXTENDED weight;
        DATA_VALUE_EXTENDED data;
    } VoxelExtended;


    typedef struct CudaTsdf {
        int side_x;
        int side_y;
        int side_z;
        int size_full;
        float side_x_meter;
        float side_y_meter;
        float side_z_meter;
        float voxel_x_meter;
        float voxel_y_meter;
        float voxel_z_meter;
        Voxel* grid;
        int block_dimension;

        CudaTsdf(int side_x, int side_y, int side_z, float side_x_meter, float side_y_meter, float side_z_meter);

        void prepareIntegration(int);
        void clear();
        void transposeMatrix(float*& matrix, float*& matrix_t, int rows, int cols);

    } CudaTsdf;
    

}


#endif /* TSDF_CUDA_H */

