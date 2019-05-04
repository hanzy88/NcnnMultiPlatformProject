cmake_minimum_required(VERSION 2.8.10)

# Create executable file
add_executable(MyProject
	Main.cpp
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
			set(NCNN_INC ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_vs2017/include)
		else()
			message(FATAL_ERROR "FATAL_ERROR: pre-built lib is not supported for the selected Visual Studio version")
		endif()
	elseif(${ARCH_TYPE} STREQUAL ARMV7)
		message("ncnn pre-built lib for ARM v7")
		set(NCNN_LIB ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/armv7/libncnn.a)  
		set(NCNN_INC ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/armv7/include)
	elseif(${ARCH_TYPE} STREQUAL LINUX)
		message("ncnn pre-built lib for PC Linux")
		set(NCNN_LIB ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_linux/libncnn.a)  
		set(NCNN_INC ${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn_prebuilt/x64_linux/include)
	endif()
else()
	message("ncnn build enabled")
	add_subdirectory(${PROJECT_SOURCE_DIR}/ExternalLibs/ncnn ncnn)
	set(NCNN_LIB ncnn)
	set(NCNN_INC ${CMAKE_CURRENT_BINARY_DIR}/ncnn/src)
endif()

target_link_libraries(MyProject ${NCNN_LIB})
target_include_directories(MyProject PUBLIC
	${NCNN_INC}
)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(MyProject PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(MyProject ${OpenCV_LIBS})
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/resource/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/resource/")
