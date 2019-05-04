cmake_minimum_required(VERSION 2.8.10)

# Create executable file
add_executable(MyProject
	Main.cpp
	${SOURCE}
)

# Set include path
target_include_directories(MyProject PUBLIC
	./
)

# For NCNN
option(USE_PREBUILT_NCNN "Use prebuilt NCNN library" ON)
if(USE_PREBUILT_NCNN)
	if(${ARCH_TYPE} STREQUAL WIN)
		if(${MSVC_VERSION} GREATER_EQUAL 1910 AND ${MSVC_VERSION} LESS 1920)
			message("ncnn pre-built lib for VS2017")
			set(NCNN_LIB
				"$<$<CONFIG:Release>:${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_vs2017/ncnn.lib>"
				"$<$<CONFIG:Debug>:${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_vs2017/ncnnd.lib>"
				"$<$<CONFIG:RelWithDebInfo>:${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_vs2017/ncnn_withd.lib>"
			)
			set(NCNN_INC ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_vs2017/generated_include)
		else()
			message(FATAL_ERROR "FATAL_ERROR: pre-built lib is not supported for the selected Visual Studio version")
		endif()
	elseif(${ARCH_TYPE} STREQUAL ARM)
		set(NCNN_LIB ${PROJECT_SOURCE_DIR}/../../modules/third_party/ncnn/prebuilt/armv7/libncnn.a)
	elseif(${ARCH_TYPE} STREQUAL LINUX)
		set(NCNN_LIB ${PROJECT_SOURCE_DIR}/../../modules/third_party/ncnn/prebuilt/x64_linux/libncnn-linux.a)  
	endif()
else()
	add_subdirectory(${PROJECT_SOURCE_DIR}/../../modules/third_party/ncnn ncnn)
	set(NCNN_LIB ncnn)
endif()

target_link_libraries(MyProject ${NCNN_LIB})
target_include_directories(MyProject PUBLIC
	${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn/src
	${NCNN_INC}
)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(MyProject PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(MyProject ${OpenCV_LIBS})
endif()


# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")
