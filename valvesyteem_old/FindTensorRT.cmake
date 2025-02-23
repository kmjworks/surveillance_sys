
find_path(TensorRT_INCLUDE_DIR NvInfer.h
          HINTS /usr/include /usr/local/include
          PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)
find_library(TensorRT_LIBRARY nvinfer
             HINTS /usr/lib /usr/local/lib
             PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)

find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin
             HINTS /usr/lib /usr/local/lib
             PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)
             
find_library(TensorRT_ONNX_LIBRARY nvonnxparser
             HINTS /usr/lib /usr/local/lib
             PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
      DEFAULT_MSG TensorRT_INCLUDE_DIR TensorRT_LIBRARY)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
  set(TensorRT_LIBRARIES    ${TensorRT_LIBRARY}
                            ${TensorRT_PLUGIN_LIBRARY}
                            ${TensorRT_ONNX_LIBRARY})
endif()
