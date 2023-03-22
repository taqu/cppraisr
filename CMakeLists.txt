cmake_minimum_required(VERSION 3.18)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME cppraisr)
project(${PROJECT_NAME})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(THIRDPARTY_ROOT ${PROJECT_ROOT}/thirdparty)

include_directories(AFTER ${PROJECT_ROOT})
include_directories(AFTER ${THIRDPARTY_ROOT})

set(OUTPUT_DIRECTORY "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

add_library(WINMMLIB MODULE IMPORTED)
set_target_properties(WINMMLIB PROPERTIES IMPORTED_LOCATION "winmm.lib")

set(HEADERS
    ${PROJECT_ROOT}/cppraisr.h
    ${PROJECT_ROOT}/util.h
    ${THIRDPARTY_ROOT}/flags/flags.h
    ${THIRDPARTY_ROOT}/stb/stb_image.h
    ${THIRDPARTY_ROOT}/stb/stb_image_write.h
    ${THIRDPARTY_ROOT}/stb/stb_image_resize.h)

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_UNICODE /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd")
    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++20")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /GL /GR- /DNDEBUG /EHsc-")

    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /SUBSYSTEM:CONSOLE")

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -std=c++20 -march=x86-64-v3")
elseif(APPLE)
endif()

add_subdirectory(train)
add_subdirectory(test)
add_subdirectory(unit_tests)

