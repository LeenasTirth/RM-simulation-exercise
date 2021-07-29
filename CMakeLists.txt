cmake_minimum_required( VERSION 2.8 )
project(Tracking)
 
set( CMAKE_CXX_FLAGS "-std=c++11 -O3")
find_package(OpenCV)
include_directories(
	${OpenCV}
	${PROJECT_SOURCE_DIR}/include)
 
add_executable(Tracking ${PROJECT_SOURCE_DIR}/src/windmill.cc ${PROJECT_SOURCE_DIR}/src/main.cpp
			${PROJECT_SOURCE_DIR}/src/WindMillDetection.cpp)
# 链接OpenCV库
target_link_libraries(Tracking ${OpenCV_LIBS})
 


