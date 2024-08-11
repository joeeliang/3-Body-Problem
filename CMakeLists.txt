cmake_minimum_required(VERSION 3.10)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)  # Update to C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the project
project(multiprocessing)

# Add include directory for fmt
include_directories(/opt/homebrew/opt/fmt/include)
include_directories(/opt/homebrew/Cellar/boost/1.85.0/include/)

# Find and include fmt library
find_package(fmt REQUIRED PATHS /opt/homebrew/opt/fmt NO_DEFAULT_PATH)

# Add the executable target
add_executable(multiprocessing long_double.cpp)

# Link the libraries to the executable target
target_link_libraries(multiprocessing PRIVATE fmt::fmt-header-only)
