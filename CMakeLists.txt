cmake_minimum_required(VERSION 2.8)
project(hpm)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_library(hpm SHARED
  src/hpm/corr.cpp
  src/hpm/feature.cpp
  src/hpm/hpmatch.cpp
)

add_executable(main
  src/draw.cpp
  src/main.cpp
)

target_link_libraries(main hpm ${OpenCV_LIBRARIES})
