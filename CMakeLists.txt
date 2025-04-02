cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(TensorrtEngine VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set paths to CUDA, and OpenCV
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4/)
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/
)

# Add the source files
add_executable(trt_inference
    main.cpp
    src/Image.cpp
    src/Inference.cpp
    src/Tensor.cpp 
    src/TensorRTEngine.cpp
)

# Link the required libraries
target_link_libraries(trt_inference
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    nvparsers
    nvinfer_plugin
    cuda
    stdc++
)
