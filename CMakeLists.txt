cmake_minimum_required(VERSION 3.15)
project(slug VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(slug src/main.cpp)

# Test executables
add_executable(slug_test_PDFs
    tests/pdfs/test_PDFs.cpp
)

# Include directories for the test executable
target_include_directories(slug_test_PDFs PRIVATE
    src
    src/pcg-cpp/include
)

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_PDFs COMMAND slug_test_PDFs)