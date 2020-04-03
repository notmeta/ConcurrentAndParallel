cmake_minimum_required(VERSION 3.16)
project(particles CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)


include_directories("/opt/cuda/include"
        "/opt/cuda/samples/common/inc/")

add_executable(particles
        kernal.cu
        main.cpp
        hitable.h
        hitable_list.h
        Ray.h
        sphere.h
        vec3.h)

set_target_properties(
        particles
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(particles -lglut -lGL -lGLU)
