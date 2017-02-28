/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CudaTsdf.h
 * Author: daniele
 *
 * Created on July 15, 2016, 12:39 PM
 */

#ifndef CUDATSDF_H
#define CUDATSDF_H

#include <slamdunk_extension/cuda/commons.hpp>

namespace slamdunk_cuda {

    class CudaTsdf {
    public:
        CudaTsdf();
        CudaTsdf(const CudaTsdf& orig);
        virtual ~CudaTsdf();

        VoxelGrid grid;
    private:

    };


    void host_set_tsdf_properties(VoxelGridProperties properties);
    void host_create_tsdf(VoxelGrid&);
    void host_clear_tsdf(VoxelGrid&);
}
#endif /* CUDATSDF_H */

