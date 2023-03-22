cmake_minimum_required(VERSION 3.18)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME test)
project(${PROJECT_NAME})

include_directories(AFTER ${PROJECT_ROOT})
include_directories(AFTER ${THIRDPARTY_ROOT})

set(OUTPUT_DIRECTORY "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

set(TEST_SOURCES
    ${PROJECT_ROOT}/cppraisr.cpp
    ${PROJECT_ROOT}/util.cpp
    test.cpp
    ${PROJECT_ROOT}/lib.cpp)

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${TEST_SOURCES})
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${HEADERS})

if(MSVC)
    target_link_libraries(${PROJECT_NAME} WINMMLIB)

elseif(UNIX)
elseif(APPLE)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")

