cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(CUDA REQUIRED)
find_package(OpenCV 4.6.0 REQUIRED COMPONENTS core dnn imgproc cudaimgproc cudabgsegm)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

set(DEEPSORT_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(DEEPSORT_INCL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/incl")

add_library(deepSort STATIC
    "${DEEPSORT_SRC_PATH}/DeepSORT.cpp"
    "${DEEPSORT_SRC_PATH}/featureTensor.cpp"
    "${DEEPSORT_SRC_PATH}/Tracker.cpp"
    "${DEEPSORT_SRC_PATH}/Track.cpp"

    "${DEEPSORT_SRC_PATH}/utilities/deepSORTEngineGenerator.cpp"
    "${DEEPSORT_SRC_PATH}/utilities/kalmanFilter.cpp"
    "${DEEPSORT_SRC_PATH}/utilities/munkres.cpp"
    "${DEEPSORT_SRC_PATH}/utilities/nnMatching.cpp"
    "${DEEPSORT_SRC_PATH}/utilities/linearAssignment.cpp"
    "${DEEPSORT_SRC_PATH}/utilities/hungarianOpener.cpp"

)

target_include_directories(deepSort PUBLIC
    "${DEEPSORT_INCL_PATH}"
    "${DEEPSORT_INCL_PATH}/utilities"
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
)


target_link_libraries(deepSort
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    Threads::Threads
    Eigen3::Eigen
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

target_compile_features(deepSort PUBLIC cxx_std_20)