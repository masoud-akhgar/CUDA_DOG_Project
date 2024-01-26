cmake_minimum_required(VERSION 3.19)
project(DoG_CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(DoG_CUDA main.cu)
target_link_libraries( DoG_CUDA ${OpenCV_LIBS} )

set_target_properties(
        DoG_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)