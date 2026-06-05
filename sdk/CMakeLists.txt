if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "can ONLY be called by parent CMakeLists.txt.")
endif()

set(VP_SDK_SOURCES
    cpp/vp_pose_pipeline.cpp
    cpp/vp_sdk_result_sink_node.cpp
    cpp/vp_sdk_frame_sink_node.cpp
)

add_library(videopipe_sdk_cpp SHARED ${VP_SDK_SOURCES})
target_link_libraries(videopipe_sdk_cpp ${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(videopipe_sdk_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(videopipe_sdk_file_sample samples/videopipe_sdk_file_sample.cpp)
target_link_libraries(videopipe_sdk_file_sample videopipe_sdk_cpp)

find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
    set(VP_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(VP_PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(VP_PYTHON_LIBRARIES ${Python3_LIBRARIES})
else()
    find_package(PythonInterp QUIET)
    find_package(PythonLibs QUIET)
    if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
        set(VP_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
        set(VP_PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
        set(VP_PYTHON_LIBRARIES ${PYTHON_LIBRARIES})
    endif()
endif()

if(VP_PYTHON_EXECUTABLE AND VP_PYTHON_INCLUDE_DIRS)
    execute_process(
        COMMAND ${VP_PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
        OUTPUT_VARIABLE VP_PYTHON_EXT_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    add_library(videopipe_sdk MODULE python/videopipe_sdk_module.cpp)
    target_link_libraries(videopipe_sdk videopipe_sdk_cpp ${VP_PYTHON_LIBRARIES})
    target_include_directories(videopipe_sdk PRIVATE ${VP_PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
    set_target_properties(videopipe_sdk PROPERTIES
        PREFIX ""
        SUFFIX "${VP_PYTHON_EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
else()
    message(WARNING "Python development files not found; skipping Python videopipe_sdk module")
endif()
