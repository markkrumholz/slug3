cmake_minimum_required(VERSION 3.15)
project(slug VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(slug src/main.cpp)

target_include_directories(slug PRIVATE
    src
    src/pcg-cpp/include
)
