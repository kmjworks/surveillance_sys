SET(NVDS_INSTALL_DIR /opt/nvidia/deepstream/deepstream)

SET(NVDS_LIBS
  nvdsgst_meta
  nvdsgst_helper
  nvdsgst_smartrecord
  nvdsgst_customhelper
  nvds_meta
  nvds_msgbroker
  nvds_utils
  nvbufsurface
  nvbufsurftransform
)



foreach(LIB ${NVDS_LIBS})
  find_library(${LIB}_PATH NAMES ${LIB} PATHS ${NVDS_INSTALL_DIR}/lib)
  if(${LIB}_PATH)
    set(NVDS_LIBRARIES ${NVDS_LIBRARIES} ${${LIB}_PATH})
  else()
    message(FATAL ERROR " Unable to find lib: ${LIB}")
    set(NVDS_LIBRARIES FALSE)
    break()
  endif()
endforeach()

find_path(NVDS_INCLUDE_DIRS
  NAMES
    nvds_version.h
  HINTS
    ${NVDS_INSTALL_DIR}/sources/includes
    ${NVDS_INSTALL_DIR}/includes
)

if (NVDS_LIBRARIES AND NVDS_INCLUDE_DIRS)
  set(NVDS_FOUND TRUE)
else()
  message(FATAL ERROR " Unable to find NVDS")
endif()

