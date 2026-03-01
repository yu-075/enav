cmake_minimum_required(VERSION 3.5)
project(map2_navigation)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/twist_to_ackermann.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  params
  maps
  DESTINATION share/${PROJECT_NAME})

ament_package()
