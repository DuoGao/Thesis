cmake_minimum_required(VERSION 2.8.3)
project(duo_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
 #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
    cv_bridge
    image_transport
    pcl_ros
    nav_msgs
    tf
    )

if(UNIX)
  add_definitions(-DUNIX)
endif(UNIX)

# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/cvlab_macros.cmake)
include(CMakeDependentOption)


# Find modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

#KDL
find_package(orocos_kdl REQUIRED)

#PCL
find_package(PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message (STATUS "Found OpenMP")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC90 OR MSVC10)
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif(OPENMP_FOUND)

#Eigen3
#find_package(Eigen3 REQUIRED)
include_directories(/usr/include/eigen3)

#OpenCV
find_package(OpenCV 2.4 REQUIRED nonfree )
if(OpenCV_INCLUDE_DIRS)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_INCLUDE_DIRS)

#BOOST
find_package(Boost 1.50.0 REQUIRED COMPONENTS random timer system)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR}) # cached

  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_definitions("-DBOOST_DISABLE_ASSERTS")
  endif()

  if(NOT BUILD_SHARED_LIBS)
    add_definitions("-DBOOST_LIB_DIAGNOSTIC -DBOOST_THREAD_USE_LIB")
  endif(NOT BUILD_SHARED_LIBS)

  if(MSVC)
    add_definitions("-DBOOST_ALL_NO_LIB")
  endif(MSVC)
else(Boost_FOUND)
  set(BOOST_ROOT "" CACHE PATH "Installation root for Boost")
  MESSAGE(FATAL_ERROR "BOOST NOT FOUND")
endif(Boost_FOUND)

#G2O
find_package(G2O)
if(G2O_FOUND)
  include_directories(${G2O_INCLUDE_DIR})
    MESSAGE("@@@@@@@@@@@@@@@@@@@@@@@@@òò G2O FOUND!")
    MESSAGE(${G2O_INCLUDE_DIR})
else(G2O_FOUND)
  set(G2O_ROOT "" CACHE PATH "Installation root for G2O")
  MESSAGE(FATAL_ERROR "G2O NOT FOUND")
endif(G2O_FOUND)

#FLANN
find_package(FLANN REQUIRED)
if(FLANN_FOUND)
  include_directories(${FLANN_INCLUDE_DIRS})
else(FLANN_FOUND)
  set(FLANN_ROOT "" CACHE PATH "Installation root for FLANN")
endif(FLANN_FOUND)

#Lar Tools
#MESSAGE("#### LAR TOOLS #####")
#find_package( )
#include_directories(${ _INCLUDE_DIRS})
#
##Lar Vision
#MESSAGE("#### LAR Vision #####")
#find_package( )
#include_directories(${ _INCLUDE_DIRS})


#### CUDA #####
if(GPU)
   MESSAGE("")
   MESSAGE("###############")
   MESSAGE("#### CUDA #####")
   MESSAGE("###############")
   MESSAGE("")

   find_package(CUDA QUIET REQUIRED)
   ### choosing the right compute capability
    ### we just start from version 1.3 ( 1.0 - 1.2 is not supported )
    list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35;-std=c++11;-O2;-DVERBOSE")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    #CUDA_ADD_EXECUTABLE(cpp11 main.cpp test.h test.cu)
    #SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
    #SET(LIB_TYPE STATIC)

endif(GPU)




################################################################################
################################################################################
################################################################################
################################################################################

#Includes
include_directories(
  include/
  include/slamdunk/
  include/slamdunk_extension/
  /home/lar/Desktop/workspace_duo/src/ /include/
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/include
  ${CATKIN_BUILD_PREFIX}/include
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lar_slam
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
INCLUDE_DIRS
   include/
   include/slamdunk/
   include/slamdunk_extension/
   LIBRARIES ${PROJECT_NAME}
   #CATKIN_DEPENDS cv_bridge image_transport pcl_ros roscpp rospy
)

set(slamdunk_libs
          ${OpenCV_LIBS}
          ${Boost_TIMER_LIBRARY}
          ${Boost_SYSTEM_LIBRARY}
          ${FLANN_LIBRARIES}
          ${G2O_STUFF_LIBRARY}
          ${G2O_CORE_LIBRARY}
          ${G2O_SOLVER_PCG}
          ${G2O_TYPES_SLAM3D}
          ${PCL_LIBRARIES}
    )

add_library(slamdunk
       src/slamdunk/data.cpp
       src/slamdunk/edge_se3_xyzpair.cpp
       src/slamdunk/feature_tracker.cpp
       src/slamdunk/gpu_features.cpp
       src/slamdunk/graph_backend.cpp
       src/slamdunk/graph_utils.cpp
       src/slamdunk/feature_matcher.cpp
       src/slamdunk/slam_dunk.cpp
       src/slamdunk/transformation_estimation.cpp
       src/slamdunk/internal_timers.hpp
       //
       src/slamdunk_extension/SlamDunkScene.cpp
       src/slamdunk_extension/PlaneFinder.cpp
       src/slamdunk_extension/Two_D_Map.cpp
       src/slamdunk_extension/Map_3D.cpp
      
       src/slamdunk_extension/tf_listener.cpp
       src/slamdunk_extension/Utility.cpp
)
target_link_libraries (slamdunk ${slamdunk_libs} ${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES})


#SLAM TRACKER
#add_executable(slamdunk_tracker  src/nodes/slamdunk_tracker.cpp)
#target_link_libraries(slamdunk_tracker ${catkin_LIBRARIES} slamdunk   ${PCL_LIBRARIES})

#SLAM TRACKER 2D Map
find_package(OpenCV REQUIRED core ocl)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_executable(slamdunk_tracker_2dmap  src/nodes/slamdunk_tracker_2dmap.cpp)
target_link_libraries(slamdunk_tracker_2dmap slamdunk ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

#pcd_reader
add_executable(pcd_reader  src/nodes/pcd_reader.cpp)
target_link_libraries(pcd_reader slamdunk ${OpenCV_LIBRARIES} ${catkin_LIBRARIES}    ${PCL_LIBRARIES})

#CLOUD FUSION
#add_executable(slamdunk_fusion  src/nodes/slamdunk_fusion.cpp)
#target_link_libraries(slamdunk_fusion ${catkin_LIBRARIES} slamdunk     ${PCL_LIBRARIES} )

#TESTING
#add_executable(testing  src/nodes/testing.cpp)
#target_link_libraries(testing ${catkin_LIBRARIES} slamdunk   ${PCL_LIBRARIES}  )


if(GPU)

    include_directories(
        include/slamdunk_extension/cuda/
    )
    add_library(
        tsdf_cuda
        src/slamdunk_extension/cuda/CudaTsdf.cpp
    )
    CUDA_ADD_LIBRARY(TestLib ${LIB_TYPE}
        src/slamdunk_extension/cuda/tsdf_cuda.cu
        src/helloWorld.cu
        )
    #target_link_libraries(TestLib slamdunk)
    target_link_libraries(tsdf_cuda TestLib)


    ADD_EXECUTABLE(cuda_node src/nodes/cuda_testing.cpp)
    ADD_DEPENDENCIES(cuda_node tsdf_cuda)
    TARGET_LINK_LIBRARIES(cuda_node
       ${catkin_LIBRARIES}
       ${PCL_LIBRARIES}
       tsdf_cuda
    )

    ADD_EXECUTABLE(slamdunk_tracker_gpu src/nodes/slamdunk_tracker_gpu.cpp)
    ADD_DEPENDENCIES(slamdunk_tracker_gpu  tsdf_cuda)
    TARGET_LINK_LIBRARIES(slamdunk_tracker_gpu
       ${catkin_LIBRARIES}
       slamdunk   ${PCL_LIBRARIES}
       tsdf_cuda
    )

    ADD_EXECUTABLE(slamdunk_tracker_gpu_faster src/nodes/slamdunk_tracker_gpu_faster.cpp)
    ADD_DEPENDENCIES(slamdunk_tracker_gpu_faster  tsdf_cuda)
    TARGET_LINK_LIBRARIES(slamdunk_tracker_gpu_faster
       ${catkin_LIBRARIES}
       slamdunk   ${PCL_LIBRARIES}
       tsdf_cuda
    )

endif(GPU)
