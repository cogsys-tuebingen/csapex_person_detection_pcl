cmake_minimum_required(VERSION 2.8.3)
project(csapex_pcl_person_detection)

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
)

###########
## Build ##
###########

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    SYSTEM ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_backport
    src/pcl_backport/hog.cpp
)

target_link_libraries(${PROJECT_NAME}_backport
    ${PCL_LIBRARIES}
)

add_custom_target(${PROJECT_NAME}_backport_qt_show_sources SOURCES
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

add_library(${PROJECT_NAME}_apex_node
    ${sources}
)

target_link_libraries(${PROJECT_NAME}_apex_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PROJECT_NAME}_backport
)

#############
## Install ##
#############

install(FILES plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(TARGETS ${PROJECT_NAME}_backport
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})
install(TARGETS ${PROJECT_NAME}_apex_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#############
## Testing ##
#############
