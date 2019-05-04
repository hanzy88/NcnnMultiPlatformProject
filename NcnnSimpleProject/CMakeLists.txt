cmake_minimum_required(VERSION 2.8.10)

# Create executable file
project(NcnnSimpleProject)
add_executable(NcnnSimpleProject
	Main.cpp
)

# For NCNN
set(PATH_TO_EXTERNAL_LIBS ${PROJECT_SOURCE_DIR}/../ExternalLibs)
if(WIN32)
	target_include_directories(NcnnSimpleProject PUBLIC ${PATH_TO_EXTERNAL_LIBS}/ncnn_prebuilt/x64_vs2017/include)
	target_link_libraries(NcnnSimpleProject ${PATH_TO_EXTERNAL_LIBS}/ncnn_prebuilt/x64_vs2017/ncnn.lib)
else()
	target_include_directories(NcnnSimpleProject PUBLIC ${PATH_TO_EXTERNAL_LIBS}/ncnn_prebuilt/x64_linux/include)	
	target_link_libraries(NcnnSimpleProject ${PATH_TO_EXTERNAL_LIBS}/ncnn_prebuilt/x64_linux/libncnn.a)
endif()

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(NcnnSimpleProject PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(NcnnSimpleProject ${OpenCV_LIBS})
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/../resource/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/resource/")
