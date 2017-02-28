#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>   
#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Core>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

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

const int N = 350;
const int BLOCK_DIM = 128;
const float N_meter = 2.0;
const float V_meter = N_meter / N;
const int MIN_WEIGHT = 10;
__device__ int GRID_SIZE = N*N*N;

typedef unsigned char GRID_TYPE;
typedef float DEPTH_IMAGE_TYPE;

GRID_TYPE *dev_grid;

typedef float EXPANDED_VALUE;
typedef float COMPRESSED_VALUE;
typedef float WEIGHT_VALUE;
__device__ int MAX_CONVERSION_VALUE = 1;

typedef struct {
    COMPRESSED_VALUE sdf;
    COMPRESSED_VALUE weight;
} VOXEL;

//typedef int VOXEL;

VOXEL *tsdf_grid;


// __device__ int indexFrom

__global__ void voxel_grid_fill(GRID_TYPE *g, GRID_TYPE value, bool auto_value) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (auto_value)
        g[threadId] = threadId;
    else {
        g[threadId] = value;
    }
}

__global__ void voxel_grid_set(GRID_TYPE *g, int index, GRID_TYPE increase) {
    g[index] += increase;
}

__global__ void voxel_grid_update_with_depth(GRID_TYPE *g, float* depth_image, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy, int increase) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols)return;

    //    
    float x, y, z;

    int ix = index % cols;
    int iy = index / cols;
    z = depth_image[index];
    x = (z / fx)*(ix - cx);
    y = (z / fy)*(iy - cy);

    float new_x, new_y, new_z;
    new_x = x * camera_transform[0] + y * camera_transform[1] + z * camera_transform[2];
    new_y = x * camera_transform[4] + y * camera_transform[5] + z * camera_transform[6];
    new_z = x * camera_transform[8] + y * camera_transform[9] + z * camera_transform[10];

    new_x += camera_transform[3];
    new_y += camera_transform[7];
    new_z += camera_transform[11];

    //            if (z > 0.0)
    //                printf("%f %f %f \n", new_x, new_y, new_z);
    int x_index, y_index, z_index;
    int v_index;
    x_index = N * (new_x / N_meter) + N / 2;
    y_index = N * (new_y / N_meter) + N / 2;
    z_index = N * (new_z / N_meter) + N / 2;

    if (x_index >= N || x_index < 0 || y_index >= N || y_index < 0 || z_index >= N || z_index < 0) return;

    v_index = x_index + y_index * N + z_index * N*N;

    if (v_index < GRID_SIZE && v_index >= 0) {
        int new_value = g[v_index];
        new_value += increase;

        if (new_value > 255) {
            new_value = 255;
        }
        if (new_value < 0) {
            new_value = 0;
        }
        g[v_index] = new_value;
    }

}

__device__ bool coordinates_to_index(float& x, float& y, float& z, int& ix, int& iy, int& iz, int& index) {
    ix = N * (x / N_meter) + N / 2;
    iy = N * (y / N_meter) + N / 2;
    iz = N * (z / N_meter);
    index = ix + iy * N + iz * N*N;
    return !(ix >= N || ix < 0 || iy >= N || iy < 0 || iz >= N || iz < 0) && index < GRID_SIZE && index >= 0;
}

__device__ void index_to_coordinates(int& index, float& x, float& y, float& z) {
    z = 1 * (N_meter * ((index / (N * N))) / N);
    y = 1 * (N_meter * (((index % (N * N))) / N) / N) - N_meter / 2.0;
    x = 1 * (N_meter * (((index % (N * N))) % N) / N) - N_meter / 2.0;
    z += V_meter * 0.5;
    x += V_meter * 0.5;
    y += V_meter * 0.5;
}

__device__ void transform_coordinates(float& x, float& y, float& z, float* camera_transform, float& x_t, float& y_t, float& z_t) {
    x_t = x * camera_transform[0] + y * camera_transform[1] + z * camera_transform[2];
    y_t = x * camera_transform[4] + y * camera_transform[5] + z * camera_transform[6];
    z_t = x * camera_transform[8] + y * camera_transform[9] + z * camera_transform[10];

    x_t += camera_transform[3];
    y_t += camera_transform[7];
    z_t += camera_transform[11];
}

__global__ void reset_tsdf(VOXEL *tsdf_grid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    tsdf_grid[index].sdf = 0;
    tsdf_grid[index].weight = false;
}

__device__ void pack_value(EXPANDED_VALUE& v1, COMPRESSED_VALUE& v2) {
    v2 = v1 * MAX_CONVERSION_VALUE;
}

__device__ void unpack_value(COMPRESSED_VALUE& v1, EXPANDED_VALUE& v2) {
    v2 = (EXPANDED_VALUE) v1 / (EXPANDED_VALUE) MAX_CONVERSION_VALUE;
}

__device__ bool get_tsdf_value(VOXEL *tsdf_grid, int ix, int iy, int iz, EXPANDED_VALUE& value) {
    int index = ix + iy * N + iz * N*N;
    if ((ix >= N || ix < 0 || iy >= N || iy < 0 || iz >= N || iz < 0))return false;
    COMPRESSED_VALUE v = tsdf_grid[index].sdf;
    unpack_value(v, value);
    return true;
}

__device__ bool get_tsdf_weight(VOXEL *tsdf_grid, int ix, int iy, int iz, COMPRESSED_VALUE& w) {
    int index = ix + iy * N + iz * N*N;
    if ((ix >= N || ix < 0 || iy >= N || iy < 0 || iz >= N || iz < 0))return false;
    COMPRESSED_VALUE v = tsdf_grid[index].weight;
    unpack_value(v, w);
    return true;
}

__device__ bool get_tsdf_value(VOXEL *tsdf_grid, float x, float y, float z, EXPANDED_VALUE& value) {
    int ix, iy, iz, index;
    coordinates_to_index(x, y, z, ix, iy, iz, index);
    return get_tsdf_value(tsdf_grid, ix, iy, iz, value);
}

__device__ bool get_tsdf_weight(VOXEL *tsdf_grid, float x, float y, float z, COMPRESSED_VALUE& w) {
    int ix, iy, iz, index;
    coordinates_to_index(x, y, z, ix, iy, iz, index);
    return get_tsdf_weight(tsdf_grid, ix, iy, iz, w);
}

__device__ bool interpol8_tsdf_value(VOXEL *tsdf_grid, float x, float y, float z, EXPANDED_VALUE &value) {

//    return get_tsdf_value(tsdf_grid, x, y, z, value);

    int ix, iy, iz, index;
    coordinates_to_index(x, y, z, ix, iy, iz, index);

    int bound = 2;
    if (ix >= N - bound || ix < bound || iy >= N - bound || iy < bound || iz >= N - bound || iz < bound)return false;

    float cx, cy, cz;
    index_to_coordinates(index, cx, cy, cz);

    if (x < cx) ix -= 1;
    if (y < cy) iy -= 1;
    if (z < cz) iz -= 1;

    float v, vx, vy, vz, vxy, vxz, vyz, vxyz;

    bool valid = true;
    valid &= get_tsdf_value(tsdf_grid, ix, iy, iz, v);
    valid &= get_tsdf_value(tsdf_grid, ix + 1, iy, iz, vx);
    valid &= get_tsdf_value(tsdf_grid, ix, iy + 1, iz, vy);
    valid &= get_tsdf_value(tsdf_grid, ix, iy, iz + 1, vz);
    valid &= get_tsdf_value(tsdf_grid, ix + 1, iy + 1, iz, vxy);
    valid &= get_tsdf_value(tsdf_grid, ix + 1, iy, iz + 1, vxz);
    valid &= get_tsdf_value(tsdf_grid, ix, iy + 1, iz + 1, vyz);
    valid &= get_tsdf_value(tsdf_grid, ix + 1, iy + 1, iz + 1, vxyz);
    
    float tx = ix*N - int(ix*N);//(x - cx) * (V_meter);
    float ty = iy*N - int(iy*N);//(y - cy) * (V_meter);
    float tz = iz*N - int(iz*N);//(z - cz) * (V_meter);
    
    value = v * (1 - tx) * (1 - ty) * (1 - tz) +
            vz * (1 - tx) * (1 - ty) * (tz) +
            vy * (1 - tx) * (ty) * (1 - tz) +
            vyz * (1 - tx) * (ty) * (tz) +
            vx * (tx) * (1 - ty) * (1 - tz) +
            vxz * (tx) * (1 - ty) * (tz) +
            vxy * (tx) * (ty) * (1 - tz) +
            vxyz * (tx) * (ty) * (tz);
    return valid;
}

__global__ void integrate_depth(VOXEL *tsdf_grid, float* depth_image, float* camera_transform, float* camera_transform_inv, int rows, int cols, float fx, float fy, float cx, float cy, float pos_trunc_dist, float neg_trunc_dist, float min_dist, float max_dist, bool revert = false) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= GRID_SIZE)return;
    //
    float x, y, z;
    index_to_coordinates(index, x, y, z);
    //

    float xt, yt, zt;
    transform_coordinates(x, y, z, camera_transform_inv, xt, yt, zt);
    if (zt < min_dist || zt > max_dist)return; //Point of the frustum culled

    int u = (xt * fx / zt) + cx;
    int v = (yt * fy / zt) + cy;

    if (u > cols || u < 0 || v > rows || v < 0)return; //Point of the frustum
    //    printf("Index:%d x:%f, y:%f, z:%f, u:%d ,v:%d \n",index,x,y,z,u,v);
    //
    int depth_index = v * cols + u;
    float depth = depth_image[depth_index];

    if (isnan(depth))return; //NaN depth value

    float sdf_new = depth - zt;
    if (sdf_new > pos_trunc_dist) {
        sdf_new = pos_trunc_dist;
    }
    if (sdf_new < -neg_trunc_dist) {
        return;
    }
    //
    if (sdf_new > 0)
        sdf_new /= pos_trunc_dist;
    else
        sdf_new /= neg_trunc_dist;
    //    short c_sdf_new = sdf_new * MAX_CONVERSION_VALUE;


    COMPRESSED_VALUE old_sdf = tsdf_grid[index].sdf;
    COMPRESSED_VALUE old_w = tsdf_grid[index].weight;

    COMPRESSED_VALUE new_w, new_sdf;

    if (revert) {
        new_w = -1;


    } else {
        new_w = 1;
    }
    if (old_w + new_w == 0) {
        new_sdf = 0;
    } else {
        new_sdf = (old_sdf * old_w + sdf_new * new_w) / (old_w + new_w);
    }
    //    if (revert) {
    //        printf("Deintegration: SDF %f W %f -> NEW_SDF %f NEW_W %f = %f\n",
    //                tsdf_grid[index].sdf,
    //                tsdf_grid[index].weight,
    //                sdf_new,
    //                new_w,
    //                new_sdf
    //                );
    //    }

    //    new_v = new_v > MAX_CONVERSION_VALUE ? MAX_CONVERSION_VALUE : new_v;
    //    new_v = new_v< -MAX_CONVERSION_VALUE ? -MAX_CONVERSION_VALUE : new_v;
    //        printf("%f \n",new_v);
    //    if (revert) {
    tsdf_grid[index].sdf = new_sdf;
    tsdf_grid[index].weight = old_w + new_w;
    //    } else {
    //        tsdf_grid[index].sdf += sdf_new;
    //    }

}

__global__
void render_tsdf_view(VOXEL* tsdf_grid, float* data, int rows, int cols, float* camera_transform, float min_dist, float max_dist, float fx, float fy, float cx, float cy) {
    //
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = index % cols;
    int iy = index / cols;

    float dir_x = (float) (ix - cx) / fx;
    float dir_y = (float) (iy - cy) / fy;
    float dir_z = 1.0f;
    float dir_mag = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    dir_x = dir_x / dir_mag;
    dir_y = dir_y / dir_mag;
    dir_z = dir_z / dir_mag;

    float new_dir_x, new_dir_y, new_dir_z;
    new_dir_x = dir_x * camera_transform[0] + dir_y * camera_transform[1] + dir_z * camera_transform[2];
    new_dir_y = dir_x * camera_transform[4] + dir_y * camera_transform[5] + dir_z * camera_transform[6];
    new_dir_z = dir_x * camera_transform[8] + dir_y * camera_transform[9] + dir_z * camera_transform[10];

    float p_x = camera_transform[3];
    float p_y = camera_transform[7];
    float p_z = camera_transform[11];

    float dist = min_dist;
    float delta = 1 * V_meter;

    p_x += dist * new_dir_x;
    p_y += dist * new_dir_y;
    p_z += dist * new_dir_z;

    bool first_voxel = true;
    bool collision = false;
    float previous_voxel_value = -1;
    float current_voxel_value = -1;

    int voxel_index;
    int x_index, y_index, z_index;
    int trial = 100;
    float value;

    WEIGHT_VALUE w;

    bool valid_voxel = false;
    bool strange_change = false;
    int max_iterations = 100;
    while (dist < max_dist && trial >= 0) {

        get_tsdf_weight(tsdf_grid, p_x, p_y, p_z, w);
//        if(w<100)continue;
        valid_voxel = get_tsdf_value(tsdf_grid, p_x, p_y, p_z, value);

        //        if (first_voxel && !valid_voxel) {
        //            continue;
        //        }
        //        
        //        if (!first_voxel && !valid_voxel) {
        //            strange_change = true;
        //        }else{
        //            strange_change = false;
        //        }
        if (!valid_voxel) {
            if (max_iterations-- <= 0)break;
            value = CUDART_NAN_F;
        }

        if (!first_voxel) {
            current_voxel_value = value;
            if (
                    (previous_voxel_value >= 0 && current_voxel_value < 0 && delta > 0) ||
                    (previous_voxel_value <= 0 && current_voxel_value > 0 && delta < 0) ||
                    (previous_voxel_value == CUDART_NAN_F && current_voxel_value != CUDART_NAN_F) ||
                    (previous_voxel_value != CUDART_NAN_F && current_voxel_value == CUDART_NAN_F)
                    ) {
                //&& previous_voxel_value <= 0
                //                delta = -delta / 2;
                //                if (fabs(delta) <= fabs(2 * V_meter)) {
                collision = true;
                break;
                //                }
                //                    printf("Inex %d, %d, %d \n", x_index,y_index,z_index);

            }
        }
        first_voxel = false;
        previous_voxel_value = value;

        p_x += delta * new_dir_x;
        p_y += delta * new_dir_y;
        p_z += delta * new_dir_z;

        dist += fabs(delta);
    }
//    float len = previous_voxel_value + fabs(current_voxel_value);
//    float back = (current_voxel_value / len) * V_meter;
//    dist += back;


    float d_xm;
    float d_xp;
    float d_ym;
    float d_yp;
    float d_zm;
    float d_zp;

    bool valid = true;

    valid &= interpol8_tsdf_value(tsdf_grid, p_x - V_meter, p_y, p_z, d_xm);
    valid &= interpol8_tsdf_value(tsdf_grid, p_x + V_meter, p_y, p_z, d_xp);
    valid &= interpol8_tsdf_value(tsdf_grid, p_x, p_y - V_meter, p_z, d_ym);
    valid &= interpol8_tsdf_value(tsdf_grid, p_x, p_y + V_meter, p_z, d_yp);
    valid &= interpol8_tsdf_value(tsdf_grid, p_x, p_y, p_z - V_meter, d_zm);
    valid &= interpol8_tsdf_value(tsdf_grid, p_x, p_y, p_z + V_meter, d_zp);

    float n_x = (d_xp - d_xm);
    float n_y = (d_yp - d_ym);
    float n_z = (d_zp - d_zm);
    float n_mag = sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
    n_x /= n_mag;
    n_y /= n_mag;
    n_z /= n_mag;

    float d[3];
    d[0] = 0.577350269;
    d[1] = 0.577350269;
    d[2] = 0.577350269;

    float d_dot_n = d[0] * n_x + d[1] * n_y + d[2] * n_z;
    float r[3];
    r[0] = d[0] - 2 * d_dot_n *n_x;
    r[1] = d[1] - 2 * d_dot_n *n_y;
    r[2] = d[2] - 2 * d_dot_n *n_z;

    float angle = (acos(r[0] * camera_transform[2] + r[1] * camera_transform[6] + r[2] * camera_transform[10]) + 1) / 2.0;
    if (fabs(angle) > 3.14)angle = 3.14 * (angle / fabs(angle));
    float mag = fabs(angle) / 3.14;

    //    printf("Ange %f\n",angle);
    if (collision && w && valid) {
        data[index] = 0.4 * angle;
    } else {
        data[index] = 0;
    }
    //        data[index] = 100;
    //    printf("Data %d -> %f\n", index, data[index]);


}

__global__
void render_voxel_grid_view(GRID_TYPE* voxel_grid_dev, float* data, int rows, int cols, float* camera_transform, float min_dist, float max_dist, float fx, float fy, float cx, float cy) {
    //
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = index % cols;
    int iy = index / cols;

    float dir_x = (float) (ix - cx) / fx;
    float dir_y = (float) (iy - cy) / fy;
    float dir_z = 1.0f;
    float dir_mag = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    dir_x = dir_x / dir_mag;
    dir_y = dir_y / dir_mag;
    dir_z = dir_z / dir_mag;

    float new_dir_x, new_dir_y, new_dir_z;
    new_dir_x = dir_x * camera_transform[0] + dir_y * camera_transform[1] + dir_z * camera_transform[2];
    new_dir_y = dir_x * camera_transform[4] + dir_y * camera_transform[5] + dir_z * camera_transform[6];
    new_dir_z = dir_x * camera_transform[8] + dir_y * camera_transform[9] + dir_z * camera_transform[10];

    float p_x = camera_transform[3];
    float p_y = camera_transform[7];
    float p_z = camera_transform[11];

    float dist = min_dist;
    float delta = 0.5 * N_meter / N;

    p_x += dist * new_dir_x;
    p_y += dist * new_dir_y;
    p_z += dist * new_dir_z;

    bool first_voxel = true;
    bool collision = false;
    GRID_TYPE previous_voxel_value = -1;
    GRID_TYPE current_voxel_value = -1;

    int voxel_index;
    int x_index, y_index, z_index;

    while (dist < max_dist) {

        x_index = N * (p_x / N_meter) + N / 2;
        y_index = N * (p_y / N_meter) + N / 2;
        z_index = N * (p_z / N_meter) + N / 2;

        voxel_index = x_index + y_index * N + z_index * N*N;

        if (voxel_index < GRID_SIZE && index >= 0) {
            if (!first_voxel) {
                current_voxel_value = voxel_grid_dev[voxel_index];
                if (current_voxel_value >= MIN_WEIGHT) {
                    //&& previous_voxel_value <= 0
                    collision = true;
                    //                    printf("Inex %d, %d, %d \n", x_index,y_index,z_index);

                    break;
                }
            }
            first_voxel = false;
            previous_voxel_value = voxel_grid_dev[voxel_index];
        }

        p_x += delta * new_dir_x;
        p_y += delta * new_dir_y;
        p_z += delta * new_dir_z;

        dist += fabs(delta);
    }
    //    if (collision)
    //        printf("Ray: %f, %f, %f (%f, %f, %f) = %f \n", p_x, p_y, p_z, new_dir_x, new_dir_y, new_dir_z, collision);

    if (collision) {
        data[index] = dist / N_meter;
    } else {
        data[index] = 0;
    }
    //        data[index] = 100;
    //    printf("Data %d -> %f\n", index, data[index]);


}

int get_grid_size() {
    return N * N*N;
}

int get_grid_side() {
    return N;
}

float get_grid_meter() {
    return N_meter;
}

void create_grid() {
    int size = N * N * N;
    cudaMalloc((void**) &dev_grid, size * sizeof (GRID_TYPE));
    cudaCheckError();
}

void create_tsdf_grid() {
    int size = N * N * N;
    SAFE_CALL(cudaMalloc((void**) &tsdf_grid, size * sizeof (VOXEL)));
    printf("TSDF created\n");
}

void clear_tsdf_grid() {
    dim3 blockSize(BLOCK_DIM);
    dim3 gridSize((int) ceil((N * N * N) / BLOCK_DIM));
    reset_tsdf << <gridSize, blockSize>>>(tsdf_grid);
    cudaCheckError();
    printf("TSDF erased\n");
}

void delete_tsdf_grid() {
    SAFE_CALL(cudaFree(tsdf_grid));
}

void delete_grid() {
    cudaFree(dev_grid);
}

void fill_grid(GRID_TYPE value = 0, bool auto_value = true) {
    dim3 blockSize(BLOCK_DIM);
    dim3 gridSize((int) ceil((N * N * N) / BLOCK_DIM));
    voxel_grid_fill << <gridSize, blockSize>>>(dev_grid, value, auto_value);
    cudaCheckError();
}

void update_voxel_grid(float* image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy, int increase) {

    int size = rows*cols;
    float* dev_image;
    SAFE_CALL(cudaMalloc((void**) &dev_image, size * sizeof (float)));
    SAFE_CALL(cudaMemcpy(dev_image, image_data, size * sizeof (float), cudaMemcpyHostToDevice));

    float* camera_transform_t = new float[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            camera_transform_t[4 * i + j] = camera_transform[4 * j + i];
            //            printf("%f ",camera_transform_t[4 * i + j]);
        }
        //        printf("\n ");
    }
    float* dev_camera;
    SAFE_CALL(cudaMalloc((void**) &dev_camera, 16 * sizeof (float)));
    SAFE_CALL(cudaMemcpy(dev_camera, camera_transform_t, 16 * sizeof (float), cudaMemcpyHostToDevice));

    cudaCheckError();
    dim3 blockSize(64);
    dim3 gridSize((int) ceil((size) / 64));

    voxel_grid_update_with_depth << <gridSize, blockSize>>>(dev_grid, dev_image, dev_camera, rows, cols, fx, fy, cx, cy, increase); //Kernel invocation
    delete[] camera_transform_t;
    cudaFree(dev_image);
    cudaFree(dev_camera);
}

void renderVoxelGridView(float*& image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy) {

    float* camera_transform_t = new float[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            camera_transform_t[4 * i + j] = camera_transform[4 * j + i];
        }
    }

    int size = rows*cols;
    //    float* data = new float[size];
    //
    float* data_dev;
    SAFE_CALL(cudaMalloc((void**) &data_dev, size * sizeof (float)));
    //
    float* camera_transform_dev;
    SAFE_CALL(cudaMalloc((void**) &camera_transform_dev, 16 * sizeof (float)));
    SAFE_CALL(cudaMemcpy(camera_transform_dev, camera_transform_t, 16 * sizeof (float), cudaMemcpyHostToDevice));
    //
    int blockSize, gridSize;
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int) ceil((float) size / blockSize);
    render_voxel_grid_view << <gridSize, blockSize>>>(dev_grid, data_dev, rows, cols, camera_transform_dev, 0.3, N_meter, fx, fy, cx, cy);

    image_data = new float[size];
    SAFE_CALL(cudaMemcpy(image_data, data_dev, size * sizeof (float), cudaMemcpyDeviceToHost));


    //
    ////    cv::Mat m(height, width, CV_32F, data);
    //    image_data = new float[size];
    //    memcpy(image_data,data,size*sizeof(float));
    //
    SAFE_CALL(cudaFree(data_dev));
    SAFE_CALL(cudaFree(camera_transform_dev));
    //    delete[] data;
    delete[] camera_transform_t;
}

void increase_value(int index) {
    //    dim3 blockSize(BLOCK_DIM);
    //    dim3 gridSize((int) ceil((N * N * N) / BLOCK_DIM));
    voxel_grid_set << <1, 1 >> >(dev_grid, index, 1);
    cudaCheckError();
}

GRID_TYPE retrieve_value(int index) {
    int size = N * N * N;
    int* grid = new int[size];
    SAFE_CALL(cudaMemcpy(grid, dev_grid, size * sizeof (GRID_TYPE), cudaMemcpyDeviceToHost));
    return grid[index];
}

void retrieve_grid(GRID_TYPE* grid) {
    int size = N * N * N;
    SAFE_CALL(cudaMemcpy(grid, dev_grid, size * sizeof (GRID_TYPE), cudaMemcpyDeviceToHost));
}
////////////////////////////////////////////

void transpose_matrix(float*& matrix, float*& matrix_t) {
    matrix_t = new float[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrix_t[4 * i + j] = matrix[4 * j + i];
        }
    }
}

void host_integrate_depth(float* image_data, float* camera_transform, float* camera_transform_inv, int rows, int cols, float fx, float fy, float cx, float cy, float pos_trunc_dist, float neg_trunc_dist, float min_dist, float max_dist, bool revert = false) {

    int size = rows*cols;
    float* dev_image;
    SAFE_CALL(cudaMalloc((void**) &dev_image, size * sizeof (float)));
    SAFE_CALL(cudaMemcpy(dev_image, image_data, size * sizeof (float), cudaMemcpyHostToDevice));


    float* camera_transform_t;
    float* camera_transform_inv_t;
    transpose_matrix(camera_transform, camera_transform_t);
    transpose_matrix(camera_transform_inv, camera_transform_inv_t);




    float* dev_camera;
    float* dev_camera_inv;
    SAFE_CALL(cudaMalloc((void**) &dev_camera, 16 * sizeof (float)));
    SAFE_CALL(cudaMalloc((void**) &dev_camera_inv, 16 * sizeof (float)));
    SAFE_CALL(cudaMemcpy(dev_camera, camera_transform_t, 16 * sizeof (float), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_camera_inv, camera_transform_inv_t, 16 * sizeof (float), cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_DIM);
    dim3 gridSize((int) ceil((N * N * N) / BLOCK_DIM));

    integrate_depth << <gridSize, blockSize>>>(tsdf_grid, dev_image, dev_camera, dev_camera_inv, rows, cols, fx, fy, cx, cy, 0.05, 0.03, 0.3, 1.0, revert); //Kernel invocation
    delete[] camera_transform_t;
    delete[] camera_transform_inv_t;
    SAFE_CALL(cudaFree(dev_image));
    SAFE_CALL(cudaFree(dev_camera_inv));
    SAFE_CALL(cudaFree(dev_camera));
}

void host_render_tsdf_view(float*& image_data, float* camera_transform, int rows, int cols, float fx, float fy, float cx, float cy) {

    float* camera_transform_t = new float[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            camera_transform_t[4 * i + j] = camera_transform[4 * j + i];
        }
    }

    int size = rows*cols;
    //    float* data = new float[size];
    //
    float* data_dev;
    SAFE_CALL(cudaMalloc((void**) &data_dev, size * sizeof (float)));
    //
    float* camera_transform_dev;
    SAFE_CALL(cudaMalloc((void**) &camera_transform_dev, 16 * sizeof (float)));
    SAFE_CALL(cudaMemcpy(camera_transform_dev, camera_transform_t, 16 * sizeof (float), cudaMemcpyHostToDevice));
    //
    int blockSize, gridSize;
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int) ceil((float) size / blockSize);
    render_tsdf_view << <gridSize, blockSize>>>(tsdf_grid, data_dev, rows, cols, camera_transform_dev, 0.3, N_meter, fx, fy, cx, cy);

    image_data = new float[size];
    SAFE_CALL(cudaMemcpy(image_data, data_dev, size * sizeof (float), cudaMemcpyDeviceToHost));


    //
    ////    cv::Mat m(height, width, CV_32F, data);
    //    image_data = new float[size];
    //    memcpy(image_data,data,size*sizeof(float));
    //
    SAFE_CALL(cudaFree(data_dev));
    SAFE_CALL(cudaFree(camera_transform_dev));
    //    delete[] data;
    delete[] camera_transform_t;
}