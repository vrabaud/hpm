cmake_minimum_required(VERSION 2.8)
project(hpm)

find_package(OpenCV REQUIRED)

include_directories(src/hpm/)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_library(hpm SHARED
  src/draw.cpp
  src/hpm/corr.cpp
  src/hpm/corr.hpp
  src/hpm/feature.cpp
  src/hpm/feature.hpp
  src/hpm/hpmatch.cpp
  src/hpm/hpmatch.hpp
  src/hpm/util.hpp
)

add_executable(main src/main.cpp)

target_link_libraries(main hpm ${OpenCV_LIBRARIES})
