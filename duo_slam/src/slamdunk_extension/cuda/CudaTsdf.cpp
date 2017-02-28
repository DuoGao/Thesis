/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CudaTsdf.cpp
 * Author: daniele
 * 
 * Created on July 15, 2016, 12:39 PM
 */

#include "CudaTsdf.h"

namespace slamdunk_cuda {

    CudaTsdf::CudaTsdf() {
        host_create_tsdf(this->grid);
        host_clear_tsdf(this->grid);
    }

    CudaTsdf::CudaTsdf(const CudaTsdf& orig) {
    }

    CudaTsdf::~CudaTsdf() {
    }
}
