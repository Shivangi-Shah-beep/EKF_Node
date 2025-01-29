cmake_minimum_required(VERSION 2.8.3)
project(prob_rob_labs)

find_package(catkin REQUIRED COMPONENTS
  opencv_apps
  roslaunch
  roscpp
)


catkin_package()

# Include directories for header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Add an executable for your node
add_executable(bayes_filter_node src/bayes_filter_node.cpp)

# Link the executable with the necessary ROS libraries
target_link_libraries(bayes_filter_node
  ${catkin_LIBRARIES}
)

add_dependencies(bayes_filter_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

roslaunch_add_file_check(launch/bayes_filter.launch)

roslaunch_add_file_check(launch/jackal_and_door.launch)

catkin_python_setup()

install(DIRECTORY launch worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS bin/bayes_filter_3
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/bayes_filter_2
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/bayes_filter
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/flaky_door_opener
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/image_mean_feature_x
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
