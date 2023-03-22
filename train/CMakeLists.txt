cmake_minimum_required(VERSION 3.18)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME train)
project(${PROJECT_NAME})

set(TRAIN_SOURCES
    ${PROJECT_ROOT}/cppraisr.cpp
    ${PROJECT_ROOT}/util.cpp
    train.cpp
    ${PROJECT_ROOT}/lib.cpp)

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${TRAIN_SOURCES})
add_executable(${PROJECT_NAME} ${TRAIN_SOURCES} ${HEADERS})

if(MSVC)
    target_link_libraries(${PROJECT_NAME} WINMMLIB)

elseif(UNIX)
elseif(APPLE)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")

