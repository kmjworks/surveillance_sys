find_path(DEEPSTREAM_APP_COMMON_INCLUDE_DIR deepstream_app_version.h
    HINTS 
    /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes
)

file(GLOB DEEPSTREAM_APP_COMMON_SRCS 
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR}/../src/*.c
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR}/../src/*.cpp
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR}/../src/deepstream-yaml/*.cpp
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR}/../src/*.c
)

mark_as_advanced(DEEPSTREAM_APP_COMMON_INCLUDE_DIR)
set(DEEPSTREAM_APP_COMMON_INCLUDE_DIRS ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR})
set(DEEPSTREAM_APP_COMMON_SRCS ${DEEPSTREAM_APP_COMMON_SRCS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DeepstreamAppCommon DEFAULT_MSG
    DEEPSTREAM_APP_COMMON_INCLUDE_DIR
    DEEPSTREAM_APP_COMMON_SRCS
)
