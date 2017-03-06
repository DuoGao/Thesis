
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

