cmake_minimum_required(VERSION 3.29.6)

project(gmsh_example VERSION 1.00)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the include directory
# NOTE: I had to force CMake to look here because it defaulted to the conda build 
set(GMSH_INCLUDE_DIRS /opt/homebrew/include)

# Find the path to the gmsh library
find_library(GMSH_LIBRARIES
names gmsh
HINTS /opt/homebrew/lib
)

if (GMSH_INCLUDE_DIRS AND GMSH_LIBRARIES)
    include_directories(${GMSH_INCLUDE_DIRS}) # location to search for include files
    message(STATUS "GMSH_INCLUDE_DIRS=${GMSH_INCLUDE_DIRS}")
    set(GMSH_LIBRARIES ${GMSH_LIBRARIES})
    message(STATUS "GMSH_LIBRARIES=${GMSH_LIBRARIES}")
    list(APPEND EXTRA_LIBS ${GMSH_LIBRARIES})
else()
    message(FATAL_ERROR "GMSH not found")
endif()

# Add the executable
add_executable(fea geom.cpp)

# Link the libraries to the exectuable
target_link_libraries(fea ${EXTRA_LIBS})
