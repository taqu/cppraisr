cmake_minimum_required(VERSION 3.18)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME unit_tests)
project(${PROJECT_NAME})

set(UNITTESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(AFTER ${UNITTESTS_ROOT})

set(UNITTESTS_HEADERS
    ${HEADERS}
    ${UNITTESTS_ROOT}/catch_amalgamated.hpp)
set(UNITTESTS_SOURCES
    ${PROJECT_ROOT}/cppraisr.cpp
    ${PROJECT_ROOT}/util.cpp
    ${PROJECT_ROOT}/lib.cpp
    ${UNITTESTS_ROOT}/main.cpp
    ${UNITTESTS_ROOT}/catch_amalgamated.cpp
    ${UNITTESTS_ROOT}/test_util.cpp)

source_group("include" FILES ${UNITTESTS_HEADERS})
source_group("src" FILES ${UNITTESTS_SOURCES})
add_executable(${PROJECT_NAME} ${UNITTESTS_HEADERS} ${UNITTESTS_SOURCES})

if(MSVC)
    target_link_libraries(${PROJECT_NAME} WINMMLIB)

elseif(UNIX)
elseif(APPLE)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")

