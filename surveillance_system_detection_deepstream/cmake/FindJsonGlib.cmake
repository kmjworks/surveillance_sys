include(LibFindMacros)

libfind_pkg_check_modules(JsonGlib_PKGCONF json-glib-1.0)

find_path(JsonGlib_INCLUDE_DIR
  NAMES json-glib/json-glib.h
  HINTS ${JsonGlib_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES json-glib-1.0
)

find_library(JsonGlib_LIBRARY
  NAMES json-glib-1.0
  HINTS ${JsonGlib_PKGCONF_LIBRARY_DIRS}
)

set(JsonGlib_PROCESS_INCLUDES ${JsonGlib_INCLUDE_DIR})
set(JsonGlib_PROCESS_LIBS ${JsonGlib_LIBRARY})
libfind_process(JsonGlib)

if(JsonGlib_FOUND)
  set(JsonGlib_INCLUDE_DIRS ${JsonGlib_INCLUDE_DIR})
  set(JsonGlib_LIBRARIES ${JsonGlib_LIBRARY})
endif(JsonGlib_FOUND)

