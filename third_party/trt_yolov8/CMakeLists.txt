cmake_minimum_required(VERSION 3.10)
project(trt_yolov8 VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

# save all libs to the same directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)

# TensorRT required
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr/local/tensorRT")
endif()
include_directories("${TENSORRT_ROOT}/include")
link_directories("${TENSORRT_ROOT}/lib")
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
enable_language(CUDA)

# CUDA required
find_package(CUDA REQUIRED) 
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV required
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# build for trt_yolov8
file(GLOB SRCS "*.cpp" "src/*.cpp" "src/*.cu" "plugin/*.cu")
add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} nvinfer cudart ${OpenCV_LIBS})

# build samples for trt_yolov8
if(NOT DEFINED VP_BUILD_FROM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples)  # save all exe to 'samples'
    
    add_executable(trt_yolov8_det_test "samples/trt_yolov8_det_test.cpp")
    target_link_libraries(trt_yolov8_det_test ${PROJECT_NAME})
    
    add_executable(trt_yolov8_pose_test "samples/trt_yolov8_pose_test.cpp")
    target_link_libraries(trt_yolov8_pose_test ${PROJECT_NAME})
    
    add_executable(trt_yolov8_seg_test "samples/trt_yolov8_seg_test.cpp")
    target_link_libraries(trt_yolov8_seg_test ${PROJECT_NAME})

    add_executable(trt_yolov8_cls_test "samples/trt_yolov8_cls_test.cpp")
    target_link_libraries(trt_yolov8_cls_test ${PROJECT_NAME})

    add_executable(trt_yolov8_wts_2_engine "samples/trt_yolov8_wts_2_engine.cpp")
    target_link_libraries(trt_yolov8_wts_2_engine ${PROJECT_NAME})
endif()
