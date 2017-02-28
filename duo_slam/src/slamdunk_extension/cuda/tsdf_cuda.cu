#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>   
#include <cuda.h>
#include <cuda_runtime.h>
#include <slamdunk_extension/cuda/commons.hpp>


#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#define CUDA_BLOCK_DIMENSION 128

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define SAFE_CALL(call)                                                                                                            \
 do                                                                                                                          \
    {                                                                                                                           \
    cudaError_t err = (call);                                                                                               \
    if(cudaSuccess != err)                                                                                                  \
            {                                                                                                                       \
        fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
        cudaDeviceReset();                                                                                                  \
        exit(EXIT_FAILURE);                                                                                                 \
            }                                                                                                                       \
    }                                                                                                                           \
        while (0)

namespace slamdunk_cuda {

    __global__ void kernel_reset_tsdf(Voxel *voxels) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        voxels[index].sdf = 0;
        voxels[index].weight = 0;
        voxels[index].data = 0;
    }


    ///////////////////////////////////////////

    __device__ void device_unpack_voxel(Voxel* voxels, int index, VoxelExtended& voxel) {
        Voxel v = voxels[index];
        voxel.sdf = (SDF_VALUE_EXTENDED) v.sdf / (SDF_VALUE_EXTENDED) SDF_PACK_UNPACK_REDUCTION;
        voxel.weight = (WEIGHT_VALUE_EXTENDED) v.weight / (WEIGHT_VALUE_EXTENDED) WEIGHT_PACK_UNPACK_REDUCTION;
        voxel.data = (DATA_VALUE_EXTENDED) v.data / (DATA_VALUE_EXTENDED) DATA_PACK_UNPACK_REDUCTION;
    }

    __device__ void device_pack_voxel(Voxel* voxels, int index, VoxelExtended& voxel) {
        voxels[index].sdf = voxel.sdf*SDF_PACK_UNPACK_REDUCTION;
        voxels[index].weight = voxel.weight*WEIGHT_PACK_UNPACK_REDUCTION;
        voxels[index].data = voxel.data*DATA_PACK_UNPACK_REDUCTION;
    }


    ///////////////////////////////////////////
    ///////////////////////////////////////////
    ///////////////////////////////////////////
    ///////////////////////////////////////////

    __device__ VoxelGridProperties* device_grid_properties = NULL;

    ///////////////////////////////////////////

    void host_set_tsdf_properties(VoxelGridProperties properties) {
        device_grid_properties = new VoxelGridProperties(properties);
    }

    ///////////////////////////////////////////

    void host_create_tsdf(VoxelGrid& grid) {
        if (device_grid_properties == NULL) {
            device_grid_properties = new VoxelGridProperties();
        }
        cudaMalloc((void**) &grid.voxels, device_grid_properties->full_size * sizeof (Voxel));
        cudaCheckError();

        printf("TSDF Created \n");
    }

    ///////////////////////////////////////////

    void host_clear_tsdf(VoxelGrid& grid) {

        dim3 blockSize(CUDA_BLOCK_DIMENSION);
        dim3 gridSize((int) ceil((device_grid_properties->full_size) / CUDA_BLOCK_DIMENSION));
        kernel_reset_tsdf << <gridSize, blockSize>>>(grid.voxels);
        cudaCheckError();

        printf("TSDF erased\n");
    }

    ///////////////////////////////////////////
    
     void host_integrate_depth(VoxelGrid& grid,float* image) {
//
//        dim3 blockSize(CUDA_BLOCK_DIMENSION);
//        dim3 gridSize((int) ceil((device_grid_properties->full_size) / CUDA_BLOCK_DIMENSION));
//        kernel_reset_tsdf << <gridSize, blockSize>>>(grid.voxels);
//        cudaCheckError();
//
//        printf("TSDF erased\n");
    }

    ///////////////////////////////////////////
}
//namespace slamdunk_cuda {
//
//
//  
//
//    CudaTsdf::CudaTsdf(int side_x, int side_y, int side_z, float side_x_meter, float side_y_meter, float side_z_meter) {
//        this->side_x = side_x;
//        this->side_y = side_y;
//        this->side_z = side_z;
//        this->side_x_meter = side_x_meter;
//        this->side_y_meter = side_y_meter;
//        this->side_z_meter = side_z_meter;
//        this->voxel_x_meter = side_x_meter / side_x;
//        this->voxel_y_meter = side_y_meter / side_y;
//        this->voxel_z_meter = side_z_meter / side_z;
//        this->size_full = side_x * side_y*side_z;
//        SAFE_CALL(cudaMalloc((void**) &grid, this->size_full * sizeof (Voxel)));
//        this->block_dimension = DEFAULT_BLOCK_DIMENSION;
//    }
//
//    void CudaTsdf::prepareIntegration(int) {
//
//    }
//
//    void CudaTsdf::clear() {
//        dim3 blockSize(this->block_dimension);
//        dim3 gridSize((int) ceil((this->size_full) / this->block_dimension));
//        cu_reset_tsdf << <gridSize, blockSize>>>(this->grid);
//        cudaCheckError();
//    }
//
//    void CudaTsdf::transposeMatrix(float*& matrix, float*& matrix_t, int rows, int cols) {
//        matrix_t = new float[rows * cols];
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                matrix_t[cols * i + j] = matrix[rows * j + i];
//            }
//        }
//    }
//
//
//
//}
