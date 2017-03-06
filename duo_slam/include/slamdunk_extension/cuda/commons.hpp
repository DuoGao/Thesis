

#ifndef COMMONS_HPP
#define COMMONS_HPP

#define DEFAULT_VOXEL_GRIDE_SIDE 64
#define DEFAULT_VOXEL_GRIDE_SIDE_METER 3.0

#include "cuda.h"
#include <cuda_runtime.h>


typedef float SDF_VALUE_EXTENDED;
typedef float WEIGHT_VALUE_EXTENDED;
typedef float DATA_VALUE_EXTENDED;
typedef short SDF_VALUE_PACKED;
typedef short WEIGHT_VALUE_PACKED;
typedef short DATA_VALUE_PACKED;

const int SDF_PACK_UNPACK_REDUCTION = 32767;
const int WEIGHT_PACK_UNPACK_REDUCTION = 32767;
const int DATA_PACK_UNPACK_REDUCTION = 32767;

/**
 *
 */
typedef struct Voxel {
    SDF_VALUE_PACKED sdf;
    WEIGHT_VALUE_PACKED weight;
    DATA_VALUE_PACKED data;
} Voxel;

/**
 *
 */
typedef struct VoxelExtended {
    SDF_VALUE_EXTENDED sdf;
    WEIGHT_VALUE_EXTENDED weight;
    DATA_VALUE_EXTENDED data;
} VoxelExtended;

/**
 *
 */
typedef struct VoxelGridProperties {
    short3 size;
    float3 size_meter;
    int full_size;

    VoxelGridProperties() {
        size.x = size.y = size.z = DEFAULT_VOXEL_GRIDE_SIDE;
        size_meter.x = size_meter.y = size_meter.z = DEFAULT_VOXEL_GRIDE_SIDE_METER;
        full_size = DEFAULT_VOXEL_GRIDE_SIDE * DEFAULT_VOXEL_GRIDE_SIDE*DEFAULT_VOXEL_GRIDE_SIDE;
    }

    VoxelGridProperties(int sx, int sy, int sz, float sxm, float sym, float szm) {
        size.x = sx;
        size.y = sy;
        size.z = sz;
        size_meter.x = sxm;
        size_meter.y = sym;
        size_meter.z = szm;
        full_size = sx * sy*sz;
    }

    VoxelGridProperties(VoxelGridProperties &prop) {
        size = prop.size;
        size_meter = prop.size_meter;
        full_size = prop.full_size;
    }
} VoxelGridProperties;

/**
 *
 */
typedef struct VoxelGrid {
    Voxel* voxels;
    VoxelGridProperties properties;
} VoxelGrid;

/**
 *
 */
typedef struct CameraParams {
    float fx, fy, cx, cy;
} CameraParams;

#endif /* COMMONS_HPP */

