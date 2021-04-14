cmake_minimum_required( VERSION 3.1)
# Create Project
project( solution )
add_executable( project point_cloud.cpp )

# Set StartUp Project (Option)
# (This setting is able to enable by using CMake 3.6.0 RC1 or later.)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

# Find Packages
# Find PCL
# find_package( PCL 1.8 REQUIRED)
message(CMAKE_CXX_FLAGS "flag is ${CMAKE_CXX_FLAGS}")

find_package( PCL 1.8 REQUIRED )

message(CMAKE_CXX_FLAGS "flag is ${CMAKE_CXX_FLAGS}")

# Find OpenCV
# set( OpenCV_DIR /usr/locoal/opencv )
find_package( OpenCV REQUIRED )


SET(COMPILE_FLAGS "-std=c++14" "-O3")
add_definitions(${COMPILE_FLAGS})

if( PCL_FOUND AND OpenCV_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
#   add_definitions( ${PCL_DEFINITIONS} )

  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )
  link_directories( ${OpenCV_LIB_DIR} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries( project ${PCL_LIBRARIES} )
  target_link_libraries( project ${OpenCV_LIBS} )
endif()

