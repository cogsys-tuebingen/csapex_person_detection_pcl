cmake_minimum_required(VERSION 2.8.3)
project(person_detection_pcl)

find_package(catkin REQUIRED
    COMPONENTS
        csapex
        csapex_core_plugins
        csapex_point_cloud
        csapex_transform
        csapex_opencv
        csapex_ml
        tf
        tf_conversions
)

find_package(PCL REQUIRED COMPONENTS common io people filter)

################################################
## Declare ROS messages, services and actions ##
################################################

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        person_detection_pcl_node
        person_detection_pcl_backport
)

###########
## Build ##
###########

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    SYSTEM ${PCL_INCLUDE_DIRS}
)

add_library(person_detection_pcl_backport
    src/pcl_backport/hog.cpp
)

target_link_libraries(person_detection_pcl_backport
    ${PCL_LIBRARIES}
)

add_custom_target(person_detection_pcl_backport_qt_show_sources SOURCES
    include/person_detection_pcl/impl/ground_based_people_detection_app.hpp
    include/person_detection_pcl/impl/head_based_subcluster.hpp
    include/person_detection_pcl/impl/height_map_2d.hpp
    include/person_detection_pcl/impl/person_classifier.hpp
    include/person_detection_pcl/impl/person_cluster.hpp

    include/person_detection_pcl/ground_based_people_detection_app.h
    include/person_detection_pcl/head_based_subcluster.h
    include/person_detection_pcl/height_map_2d.h
    include/person_detection_pcl/hog.h
    include/person_detection_pcl/person_classifier.h
    include/person_detection_pcl/person_cluster.h
)


file(GLOB sources
    src/csapex/person_detection_pcl.cpp
    src/csapex/person_detection_pcl.hpp
    src/csapex/ground_plane_estimation.cpp
    src/csapex/ground_plane_estimation.hpp
    src/csapex/pcl_hog_extractor.hpp
    src/csapex/pcl_hog_extractor.cpp
)

add_library(person_detection_pcl_node
    ${sources}
)

target_link_libraries(person_detection_pcl_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    person_detection_pcl_backport
)

#############
## Install ##
#############

install(FILES plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(TARGETS person_detection_pcl_backport
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})
install(TARGETS person_detection_pcl_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#############
## Testing ##
#############
