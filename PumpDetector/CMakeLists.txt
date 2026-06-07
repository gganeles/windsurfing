cmake_minimum_required(VERSION 3.10)
project(goProPumpingParser C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Platform-specific definitions
if(WIN32)
    add_definitions(-DWIN_OS -DWIN32 -DGLEW_STATIC)
elseif(APPLE)
    add_definitions(-D__APPLE__)
elseif(UNIX AND NOT APPLE)
    add_definitions(-DLINUX)
endif()

# We want to use the internal GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

# Include directories
include_directories(
    libdspl-2.0/include
    gpmf-parser
    gpmf-parser/demo
    implot
    imgui
    imgui/backends
    glfw/include
    libtinyfiledialogs
)

# Core source files
set(SOURCES
    goProPumpingParser.cpp
    pumpDetector.cpp
    
    # GPMF
    gpmf-parser/GPMF_parser.c
    gpmf-parser/GPMF_utils.c
    gpmf-parser/demo/GPMF_mp4reader.c
    
    # ImGui
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    
    # ImPlot
    implot/implot.cpp
    implot/implot_items.cpp
    
    # TinyFileDialogs
    libtinyfiledialogs/tinyfiledialogs.c
)

add_executable(goProPumpingParser ${SOURCES})

# Target linking
target_link_libraries(goProPumpingParser PRIVATE glfw)

if(WIN32)
    target_link_libraries(goProPumpingParser PRIVATE
        opengl32 gdi32 user32 kernel32 shell32 ole32 uuid comctl32 comdlg32
        # You will also need to link the Windows version of libdspl-2.0 here if you want:
        # ${CMAKE_CURRENT_SOURCE_DIR}/libdspl-2.0/_release/libdspl.a
        # ${CMAKE_CURRENT_SOURCE_DIR}/libdspl-2.0/dspl/blas/bin/libblas.a
        # ${CMAKE_CURRENT_SOURCE_DIR}/libdspl-2.0/dspl/lapack/bin/liblapack_double.a
        # ${CMAKE_CURRENT_SOURCE_DIR}/libdspl-2.0/dspl/lapack/bin/liblapack_complex.a
    )
elseif(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_library(OPENGL_FRAMEWORK OpenGL)
    find_library(IOKIT_FRAMEWORK IOKit)
    find_library(COREVIDEO_FRAMEWORK CoreVideo)
    target_link_libraries(goProPumpingParser PRIVATE
        ${COCOA_FRAMEWORK} ${OPENGL_FRAMEWORK} ${IOKIT_FRAMEWORK} ${COREVIDEO_FRAMEWORK}
    )
elseif(UNIX AND NOT APPLE)
    find_package(OpenGL REQUIRED)
    find_package(Threads REQUIRED)
    target_link_libraries(goProPumpingParser PRIVATE
        OpenGL::GL Threads::Threads dl m
        ${CMAKE_CURRENT_SOURCE_DIR}/libdspl-2.0/_release/libdspl.so
    )
endif()

# NOTE: For libdspl-2.0, you must configure its build process or find packages for the platform.
