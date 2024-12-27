cmake_minimum_required(VERSION 3.10)
project(MyProject)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(MyProject src/rune_detector.cpp)

# 链接OpenCV库
target_link_libraries(MyProject ${OpenCV_LIBS})