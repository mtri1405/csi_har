# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.22)

# Component dung chung voi project csi_send: csi_proto.h
set(EXTRA_COMPONENT_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../shared)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(csi_recv)