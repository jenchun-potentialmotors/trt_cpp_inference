cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(TensorrtEngine VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# Find OpenCV
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# Include directories
include_directories(
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
)
