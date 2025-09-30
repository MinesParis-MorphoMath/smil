# Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Matthieu FAESSEL, or ARMINES nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.20)

project(
  Smil
  VERSION 1.1.2
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# CMAKE POLICIES ######

if(CMAKE_VERSION VERSION_GREATER 2.8.0)
  cmake_policy(SET CMP0015 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER 2.8.4)
  cmake_policy(SET CMP0017 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  cmake_policy(SET CMP0020 NEW)
endif()

# if (${CMAKE_VERSION} VERSION_GREATER 3.13.0) CMAKE_POLICY(SET CMP0078 OLD)
# endif ()

# if (${CMAKE_VERSION} VERSION_GREATER 3.14.0) CMAKE_POLICY(SET CMP0086 OLD)
# endif ()

# FORBID COMPILATION IN SOURCE DIR ######
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(
    FATAL_ERROR
      "Compilation in source directory is deprecated and will result in build errors. Please build in a different directory."
  )
endif(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

# Generate compilation database to be used by external tools like clang-tidy or
# rtags
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# ##############################################################################

set(MODULES Core Gui Base IO Morpho Advanced)

set(SMIL_VERSION ${PROJECT_VERSION})
add_compile_definitions(SMIL_MAJOR_VERSION=${PROJECT_VERSION_MAJOR})
add_compile_definitions(SMIL_MINOR_VERSION=${PROJECT_VERSION_MINOR})
add_compile_definitions(SMIL_PATCH_VERSION=${PROJECT_VERSION_PATCH})
add_compile_definitions(SMIL_VERSION=${PROJECT_VERSION})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake"
     "${CMAKE_CURRENT_SOURCE_DIR}/CMake/swig" ${PROJECT_BINARY_DIR})

include_directories(${CMAKE_MODULE_PATH})
include(GenMacros)

set(SMIL_LIB_PREFIX smil)
set(SMIL_EXT_DEPS)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND PREFIX)
  set(CMAKE_INSTALL_PREFIX
      ${PREFIX}
      CACHE FORCE)
endif()

# ADDITIONAL SEARCH PATHS (Usefull for windows and virtual environnments)
set(ADDITIONAL_SEARCH_PATHS
    ""
    CACHE STRING "Additional paths to search libraries/files")
# IF(UNIX) MARK_AS_ADVANCED(ADDITIONAL_SEARCH_PATHS) ENDIF(UNIX)
if(ADDITIONAL_SEARCH_PATHS)
  foreach(_PATH ${ADDITIONAL_SEARCH_PATHS})
    list(APPEND CMAKE_PREFIX_PATH ${_PATH} ${_PATH}/bin)
  endforeach(_PATH ${ADDITIONAL_SEARCH_PATHS})
endif(ADDITIONAL_SEARCH_PATHS)

# COMPILATION PATHS ######
set(IMAGESDIR ${PROJECT_SOURCE_DIR}/images)
configure_file("${PROJECT_SOURCE_DIR}/CMake/Smil-build.h.in"
               "${PROJECT_BINARY_DIR}/Smil-build.h")

if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH
      ${PROJECT_BINARY_DIR}/lib
      CACHE PATH "Single output directory for building all libraries.")
  mark_as_advanced(LIBRARY_OUTPUT_PATH)
endif(NOT LIBRARY_OUTPUT_PATH)
link_directories(${LIBRARY_OUTPUT_PATH})

if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH
      ${PROJECT_BINARY_DIR}/bin
      CACHE PATH "Single output directory for building all binaries.")
  mark_as_advanced(EXECUTABLE_OUTPUT_PATH)
endif(NOT EXECUTABLE_OUTPUT_PATH)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

# INSTALL PATHS ######

if(WIN32)
  set(SMIL_LIBRARIES_INSTALL_PATH lib)
  set(SMIL_HEADERS_INSTALL_PATH include)
  set(SMIL_SHARE_INSTALL_PATH share)
endif(WIN32)

if(UNIX)
  set(LIBSUFFIX)
  execute_process(
    COMMAND grep ^ID= /etc/os-release
    COMMAND sed -e "s/\"//g"
    OUTPUT_VARIABLE OSID)
  # MESSAGE(STATUS "OS_ID : ${OSID}")
  if(${OSID} MATCHES "(centos|fedora)")
    # message(STATUS "Found it " ${OSID})
    if(EXISTS "/usr/lib64" AND NOT IS_SYMLINK "/usr/lib64")
      # Smil libraries should go to /usr/local/lib64
      set(LIBSUFFIX 64)
    endif()
  endif()

  message(STATUS "LIBSUFFIX " ${LIBSUFFIX})

  # IF(EXISTS "/usr/lib64" AND NOT IS_SYMLINK "/usr/lib64") # Smil libraries
  # should go to /usr/local/lib64 SET(LIBSUFFIX 64) #SET(LIBSUFFIX) ELSE() #
  # Smil libraries should go to /usr/local/lib SET(LIBSUFFIX) ENDIF(EXISTS
  # "/usr/lib64" AND NOT IS_SYMLINK "/usr/lib64")

  set(SMIL_LIBRARIES_INSTALL_PATH
      ${CMAKE_INSTALL_PREFIX}/lib${LIBSUFFIX}/${CMAKE_PROJECT_NAME})
  set(SMIL_HEADERS_INSTALL_PATH
      ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME})
  set(SMIL_SHARE_INSTALL_PATH
      ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME})
endif(UNIX)

set(CMAKE_INSTALL_RPATH ${LIBRARY_OUTPUT_PATH})

# BUILD OPTIONS ######

# Default build type : Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_TOOLCHAIN_FILE)
  if(BUILD_SHARED_LIBS)
    option(FORCE_BUILD_SHARED_LIBS_OFF "" ON)
  else(BUILD_SHARED_LIBS)
    option(FORCE_BUILD_SHARED_LIBS_ON "" OFF)
  endif(BUILD_SHARED_LIBS)
  if(FORCE_BUILD_SHARED_LIBS_OFF)
    set(BUILD_SHARED_LIBS
        OFF
        CACHE BOOL "" FORCE)
  elseif(FORCE_BUILD_SHARED_LIBS_ON)
    set(BUILD_SHARED_LIBS
        ON
        CACHE BOOL "" FORCE)
  endif(FORCE_BUILD_SHARED_LIBS_OFF)
else(CMAKE_TOOLCHAIN_FILE)
  option(BUILD_SHARED_LIBS "Build shared libs" OFF)
endif(CMAKE_TOOLCHAIN_FILE)
mark_as_advanced(BUILD_SHARED_LIBS)

if(BUILD_SHARED_LIBS)
  add_compile_definitions(SHARED_LIBS)
  set(BUILD_SHARED_LIBS
      ON
      CACHE BOOL "" FORCE)
endif(BUILD_SHARED_LIBS)

option(USE_STATIC_DEPEND_LIBS
       "Use the static version on dependency libraries. Requires pkg-config."
       OFF)
mark_as_advanced(USE_STATIC_DEPEND_LIBS)
if(USE_STATIC_DEPEND_LIBS)
  add_link_options(-Bstatic)
endif(USE_STATIC_DEPEND_LIBS)

# 64 Bits
if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8" OR "${CMAKE_GENERATOR}" MATCHES
                                            "(Win64|IA64)")
  # 4 -> 32 BITS, 8 -> 64 BITS
  message(STATUS "64 bits system detected")
  option(USE_64BIT_IDS "Compile for 64 bits system" ON)
  mark_as_advanced(USE_64BIT_IDS)
  if(USE_64BIT_IDS)
    add_compile_definitions(USE_64BIT_IDS)
    if(NOT WIN32)
      add_compile_options(-fPIC)
    endif(NOT WIN32)
  else(USE_64BIT_IDS)
    add_compile_options(-m32)
  endif(USE_64BIT_IDS)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(STATUS "32 bits system detected")
endif()

if(ADDITIONAL_INCLUDE_PATHS)
  include_directories(${ADDITIONAL_INCLUDE_PATHS})
endif(ADDITIONAL_INCLUDE_PATHS)

# Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

# Windows 32 and 64 Bits
if(WIN32)
  if(USE_64BIT_IDS)
    # Necessary in particular for python 64 bits (PyInitModule4)
    add_compile_definitions(MS_WIN64)
  endif(USE_64BIT_IDS)
endif(WIN32)

if(MINGW)
  # Problemes de link avec Mingw (en particulier pour imports de Java JNI)
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--kill-at")
endif(MINGW)

if(MSVC)
  add_compile_options(/wd4267 /EHsc)
endif(MSVC)

# OPTIMIZATIONS ######

option(USE_OPTIMIZATION "Use optimizations" ON)
if(USE_OPTIMIZATION)
  set(TARGET_ARCHITECTURE
      "auto"
      CACHE
        STRING
        "CPU architecture to optimize for. Using an incorrect setting here can result in crashes of the resulting binary because of invalid instructions used.\nSetting the value to \"auto\" will try to optimize for the architecture where cmake is called.\nOther supported values are: \"none\", \"generic\", \"core\", \"merom\" (65nm Core2), \"penryn\" (45nm Core2), \"nehalem\", \"westmere\", \"sandy-bridge\", \"atom\", \"k8\", \"k8-sse3\", \"barcelona\", \"istanbul\", \"magny-cours\", \"bulldozer\", \"interlagos\"."
  )

  option(VERBOSE_OPTIMIZATION "Show verbose optimizations" OFF)
  mark_as_advanced(VERBOSE_OPTIMIZATION)

  if(CMAKE_COMPILER_IS_GNUCC)
    if(TARGET_ARCHITECTURE STREQUAL "auto")
      add_compile_options(-march=native -mtune=generic)
    else()
      add_compile_options(-march=${TARGET_ARCHITECTURE} -mtune=generic)
    endif()

    if(VERBOSE_OPTIMIZATION)
      add_compile_options(-ftree-vectorizer-verbose=1)
    endif()

  endif()
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT WIN32)
  find_package(OpenMP)
endif()
if(OpenMP_FOUND)
  option(USE_OPEN_MP "Use OpenMP parallelization" ON)
else()
  option(USE_OPEN_MP "Use OpenMP parallelization" OFF)
endif()
if(USE_OPEN_MP)
  find_package(OpenMP REQUIRED)
  add_compile_definitions(USE_OPEN_MP)
  link_libraries(OpenMP::OpenMP_CXX)
  if(CMAKE_COMPILER_IS_GNUCXX OR MINGW)
    list(APPEND SMIL_EXT_DEPS gomp)
    if(MINGW)
      list(APPEND SMIL_EXT_DEPS pthread)
      add_compile_options(-mstackrealign)
    endif(MINGW)
  endif(CMAKE_COMPILER_IS_GNUCXX OR MINGW)
  if(CMAKE_COMPILER_IS_CLANGXX)
    list(APPEND SMIL_EXT_DEPS omp)
  endif(CMAKE_COMPILER_IS_CLANGXX)
endif(USE_OPEN_MP)

#
# WRAPPING ###
#
option(WRAP_CPP "Wrap non-template C++ headers and libraries" FALSE)
if(WRAP_CPP)
  include(WrapCpp)
  find_package(Python3 REQUIRED)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
endif(WRAP_CPP)

option(WRAP_PYTHON "Wrap Python" FALSE)
if(WRAP_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development.Embed)
  link_libraries(Python3::Python)

  add_custom_target(python)
endif(WRAP_PYTHON)

#
# As we're wrapping to Python, let's find Numpy includes dir
#
# IF(WRAP_PYTHON AND NOT Python_Numpy_INCLUDE_DIRS)
if(WRAP_PYTHON)
  option(USE_NUMPY "Use Numpy for python pixels manipulation." OFF)

  if(USE_NUMPY)
    find_package(Python3 REQUIRED COMPONENTS NumPy)
    add_compile_definitions(USE_NUMPY)
    link_libraries(Python3::NumPy)
    message(STATUS "Numpy includes: " ${Python3_NumPy_INCLUDE_DIRS})
  endif(USE_NUMPY)
endif(WRAP_PYTHON)

#
option(WRAP_JAVA "Wrap Java" FALSE)
if(WRAP_JAVA)
  find_package(JNI)
  if(NOT JAVA_INCLUDE_PATH)
    message(SEND_ERROR "Could not find JAVA_INCLUDE_PATH containing jni.h ")
  endif(NOT JAVA_INCLUDE_PATH)
  include_directories(${JAVA_INCLUDE_PATH})
  add_custom_target(java)
endif(WRAP_JAVA)

#
option(WRAP_OCTAVE "Wrap Octave" FALSE)
if(WRAP_OCTAVE)
  find_path(OCTAVE_INCLUDE_DIR "octave/oct.h" Path to octave/oct.h)
  include_directories(${OCTAVE_INCLUDE_DIR} ${OCTAVE_INCLUDE_DIR}/octave)
  # Octave now requires mpi headers
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  add_custom_target(octave)
endif(WRAP_OCTAVE)

option(WRAP_RUBY "Wrap Ruby" FALSE)
if(WRAP_RUBY)
  find_package(Ruby REQUIRED)
  include_directories(${RUBY_INCLUDE_PATH})
  add_custom_target(ruby)
endif(WRAP_RUBY)

option(WRAP_PHP "Wrap Php" FALSE)
if(WRAP_PHP)
  find_package(PHP5 REQUIRED)
  include_directories(${PHP5_INCLUDE_PATH} ${LIBRARY_OUTPUT_PATH})
  add_custom_target(php)
endif(WRAP_PHP)

#
if(WRAP_CPP
   OR WRAP_PYTHON
   OR WRAP_JAVA
   OR WRAP_OCTAVE
   OR WRAP_RUBY
   OR WRAP_PHP)

  set(USE_WRAPPER TRUE)

  if(CMAKE_CROSSCOMPILING)
    # Problems with FindSWIG when cross-compiling...
    list(APPEND CMAKE_MODULE_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMake/CrossCompilation")
    if(USE_64BIT_IDS)
      add_compile_definitions(MS_WIN64)
    endif(USE_64BIT_IDS)
    # CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/CMake/_FindSWIG.cmake
    # ${PROJECT_BINARY_DIR}/FindSWIG.cmake) LIST(APPEND CMAKE_FIND_ROOT_PATH
    # /usr/local/mingw32)
  endif(CMAKE_CROSSCOMPILING)

  find_package(SWIG 4.0 REQUIRED)
  message(STATUS "Swig version : ${SWIG_VERSION}")
  include(${SWIG_USE_FILE})

  set(CMAKE_SWIG_OUTDIR ${LIBRARY_OUTPUT_PATH})
endif(
  WRAP_CPP
  OR WRAP_PYTHON
  OR WRAP_JAVA
  OR WRAP_OCTAVE
  OR WRAP_RUBY
  OR WRAP_PHP)

# GUI ######

find_package(Qt5Widgets)

if(Qt5Core_FOUND)
  option(USE_QT "Build Qt Gui" ON)
else()
  option(USE_QT "Build Qt Gui" OFF)
endif()

if(USE_QT)
  set(USE_QT_VERSION
      "5"
      CACHE STRING "Qt version to use")

  if(CMAKE_CROSSCOMPILING)
    set(QT_HEADERS_DIR "/usr/i686-w64-mingw32/include")
    set(QT_LIBRARY_DIR "/usr/i686-w64-mingw32/lib")
  endif()
  set(QT_USE_FRAMEWORKS
      OFF
      CACHE INTERNAL "" FORCE)

  find_package(Qt5Widgets REQUIRED)
  # SET(CMAKE_AUTOMOC ON) SET(CMAKE_AUTOUIC ON)
  link_libraries(Qt5::Core Qt5::Gui Qt5::Widgets)
  list(APPEND SMIL_EXT_DEPS Qt5::Core Qt5::Gui Qt5::Widgets)

  add_compile_definitions(USE_QT)

  if(USE_STATIC_DEPEND_LIBS)
    add_pkg_config_defs(QtCore STATIC)
    add_pkg_config_defs(QtGui STATIC)
  endif(USE_STATIC_DEPEND_LIBS)

  find_package(Qwt)
  if(QWT_FOUND)
    option(USE_QWT "Use Qwt for Qt viewer" ON)
  else(QWT_FOUND)
    option(USE_QWT "Use Qwt for Qt viewer" OFF)
  endif(QWT_FOUND)
  if(USE_QWT)
    find_package(Qwt REQUIRED)
    if(QWT_FOUND)
      add_compile_definitions(USE_QWT)
      if(MSVC)
        add_compile_definitions(_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS)
        add_compile_definitions(QWT_DLL)
      endif(MSVC)
      include_directories(${QWT_INCLUDE_DIR})
      list(APPEND SMIL_EXT_DEPS ${QWT_LIBRARY})
    endif(QWT_FOUND)
  endif(USE_QWT)

endif(USE_QT)

option(USE_AALIB "Build AALib Gui" OFF)
if(USE_AALIB)
  find_library(AALIB_LIBRARY aa)
  find_path(AALIB_INCLUDES aalib.h)
  add_compile_definitions(USE_AALIB)
  include_directories(${AALIB_INCLUDES})
  link_libraries(${AALIB_LIBRARY})
  list(APPEND SMIL_EXT_DEPS ${AALIB_LIBRARY})
endif(USE_AALIB)

# IO ######

find_package(PNG)
if(PNG_FOUND)
  option(USE_PNG "Use PNG" ON)
else(PNG_FOUND)
  option(USE_PNG "Use PNG" OFF)
endif(PNG_FOUND)
if(USE_PNG)
  add_compile_definitions(USE_PNG)
  find_package(PNG REQUIRED)
  link_libraries(PNG::PNG)
  list(APPEND SMIL_EXT_DEPS ${PNG_LIBRARIES})
endif(USE_PNG)

find_package(JPEG)
if(JPEG_FOUND)
  option(USE_JPEG "Use JPEG" ON)
else(JPEG_FOUND)
  option(USE_JPEG "Use JPEG" OFF)
endif(JPEG_FOUND)
if(USE_JPEG)
  add_compile_definitions(USE_JPEG)
  find_package(JPEG REQUIRED)
  link_libraries(JPEG::JPEG)
  list(APPEND SMIL_EXT_DEPS ${JPEG_LIBRARIES})
endif(USE_JPEG)

find_package(TIFF)
if(TIFF_FOUND)
  option(USE_TIFF "Use TIFF IO library" ON)
else(TIFF_FOUND)
  option(USE_TIFF "Use TIFF IO library" OFF)
endif(TIFF_FOUND)
if(USE_TIFF)
  add_compile_definitions(USE_TIFF)
  find_package(TIFF REQUIRED)
  link_libraries(TIFF::TIFF)
  list(APPEND SMIL_EXT_DEPS ${TIFF_LIBRARIES})
endif(USE_TIFF)

find_package(CURL)
if(CURL_FOUND)
  option(USE_CURL "Use Curl for socket requests" ON)
else(CURL_FOUND)
  option(USE_CURL "Use Curl for socket requests" OFF)
endif(CURL_FOUND)
if(USE_CURL)
  find_package(CURL REQUIRED)
  add_compile_definitions(USE_CURL)
  link_libraries(CURL::libcurl)

  list(APPEND SMIL_EXT_DEPS ${CURL_LIBRARIES})

  if(USE_STATIC_DEPEND_LIBS)
    # Find Curl pkg-config entries
    add_pkg_config_defs(libcurl STATIC)
  endif(USE_STATIC_DEPEND_LIBS)
endif(USE_CURL)

# MISC ######
find_package(Freetype)
if(FREETYPE_FOUND)
  option(USE_FREETYPE "Use FreeType library for text drawing" ON)
else(FREETYPE_FOUND)
  option(USE_FREETYPE "Use FreeType library for text drawing" OFF)
endif(FREETYPE_FOUND)

if(USE_FREETYPE)
  find_package(Freetype REQUIRED)
  add_compile_definitions(USE_FREETYPE)
  link_libraries(Freetype::Freetype)
  list(APPEND SMIL_EXT_DEPS ${FREETYPE_LIBRARY})
endif(USE_FREETYPE)

# DOC ######

option(BUILD_DOC "Generate documentation" OFF)

# TESTS ######

option(BUILD_TEST "Build tests" OFF)
if(BUILD_TEST)
  enable_testing()
  add_custom_target(tests)
  add_custom_target(benchs)
  add_subdirectory(doc/demos/python)
endif(BUILD_TEST)

# IMAGE TYPES WRAPPED ######

set(IMAGE_TYPES
    "UINT8;UINT16;UINT32"
    CACHE STRING "Image types wrapped (first type will be the default one)")
set(IMAGE_TYPES_SUPPL
    "RGB"
    CACHE
      STRING
      "Supplementary image types for base operations (read/write, copy, ...)")
list(GET IMAGE_TYPES 0 DEFAULT_IMAGE_TYPE)

set(IMAGE_SPECIAL_TYPES)
set(DATA_TYPES_STR)
set(DATA_TYPES_QUOTE_STR)
set(IMAGE_TYPES_STR)

set(SWIG_DEPS smilCore)

foreach(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NSTypes/${_IMG_TYPE})
    list(APPEND IMAGE_SPECIAL_TYPES ${_IMG_TYPE})
    list(APPEND SWIG_DEPS smil${_IMG_TYPE})
  endif(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NSTypes/${_IMG_TYPE})
  string(TOUPPER ${_IMG_TYPE} _IMG_TYPE_UP)
  add_compile_definitions(SMIL_WRAP_${_IMG_TYPE_UP})
  set(WRAP_${_IMG_TYPE_UP} ON)
  set(DATA_TYPES_QUOTE_STR ${DATA_TYPES_QUOTE_STR}"${_IMG_TYPE}",)
  set(IMAGE_TYPES_STR ${IMAGE_TYPES_STR}Image_${_IMG_TYPE},)
endforeach(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})

foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(DATA_TYPES_STR ${DATA_TYPES_STR}${_IMG_TYPE},)
endforeach(_IMG_TYPE ${IMAGE_TYPES})

string(REGEX REPLACE ",$" "" IMAGE_TYPES_STR ${IMAGE_TYPES_STR})
string(REGEX REPLACE ",$" "" DATA_TYPES_STR ${DATA_TYPES_STR})
string(REGEX REPLACE ",$" "" DATA_TYPES_QUOTE_STR ${DATA_TYPES_QUOTE_STR})

if(IMAGE_TYPES_SUPPL)
  set(DATA_TYPES_STR "${DATA_TYPES_STR} (+")
  foreach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
    set(DATA_TYPES_STR ${DATA_TYPES_STR}${_IMG_TYPE},)
  endforeach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
  string(REGEX REPLACE ",$" ")" DATA_TYPES_STR ${DATA_TYPES_STR})
endif(IMAGE_TYPES_SUPPL)

include(SwigMacros)

message(STATUS "Image types generated: ${DATA_TYPES_STR}")

# Retrieve all compile flags
get_directory_property(COMP_DEFS COMPILE_DEFINITIONS)
set(COMPILE_FLAGS)
foreach(_DEF ${COMP_DEFS})
  set(COMPILE_FLAGS ${COMPILE_FLAGS} -D${_DEF})
endforeach(_DEF ${COMP_DEFS})

if(USE_WRAPPER)
  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} ${COMPILE_FLAGS})
  # SWIG runtime code (see http://www.swig.org/Doc1.3/Modules.html#Modules_nn2)
  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} -O -relativeimport)
endif(USE_WRAPPER)

set(SMIL_SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
set(SMIL_LIBS)
set(SWIG_INTERFACE_FILES)
set(SMIL_GLOBAL_HEADERS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})
  set(_SPEC_TYPE_DIR "NSTypes/${_IMG_TYPE}")
  include_directories(${_SPEC_TYPE_DIR} ${_SPEC_TYPE_DIR}/include)
endforeach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})

foreach(_MOD ${MODULES})
  set(SMIL_SRC_DIRS
      ${SMIL_SRC_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/${_MOD}
      ${CMAKE_CURRENT_SOURCE_DIR}/${_MOD}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/${_MOD}/include/private
      ${CMAKE_CURRENT_SOURCE_DIR}/${_MOD}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/${_MOD}/test)
endforeach(_MOD ${MODULES})

if(BUILD_DOC)
  add_subdirectory(doc)
  # FILE(GLOB DOXY_SWIG_FILES ${CMAKE_CURRENT_BINARY_DIR}/doc/xml/*.i)
  # SET(SWIG_INTERFACE_FILES ${SWIG_INTERFACE_FILES} ${DOXY_SWIG_FILES})
  # MESSAGE(${SWIG_INTERFACE_FILES})
endif(BUILD_DOC)

include_directories(${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

foreach(_MOD ${MODULES})
  include_directories(${_MOD})
  # INCLUDE_DIRECTORIES(${_MOD} ${_MOD}/include ${_MOD}/include/private)
  add_subdirectory(${_MOD})
endforeach(_MOD ${MODULES})

# Special Types
foreach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})
  set(_SPEC_TYPE_DIR "NSTypes/${_IMG_TYPE}")
  add_subdirectory(${_SPEC_TYPE_DIR})
endforeach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})

# ADDONS ######

add_subdirectory(Addons)

# USER MODULES ######

set(ADD_MOD_SUBDIRS
    "UserModules"
    CACHE STRING "Additional module subdirectories")

set(AVAILABLE_USER_MODULES)

foreach(_MOD_SUB_DIR ${ADD_MOD_SUBDIRS})
  add_subdirectory(${_MOD_SUB_DIR})
endforeach(_MOD_SUB_DIR ${ADD_MOD_SUBDIRS})

if(USE_WRAPPER)
  set(SWIG_MAIN_INTERFACE ${CMAKE_BINARY_DIR}/smil.i)
  set(SWIG_COMMON_INTERFACE ${CMAKE_BINARY_DIR}/smilCommon.i)

  set(SWIG_COMMON_INCLUDES)
  if(EXISTS ${PROJECT_BINARY_DIR}/smilDoc.i)
    set(SWIG_COMMON_INCLUDES "${SWIG_COMMON_INCLUDES}%include \"smilDoc.i\"\n")
  endif(EXISTS ${PROJECT_BINARY_DIR}/smilDoc.i)
  set(SWIG_COMMON_INCLUDES "${SWIG_COMMON_INCLUDES}%{\n")
  foreach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})
    set(SWIG_COMMON_INCLUDES
        "${SWIG_COMMON_INCLUDES}#include \"NSTypes/${_IMG_TYPE}/include/D${_IMG_TYPE}.h\"\n"
    )
  endforeach(_IMG_TYPE ${IMAGE_SPECIAL_TYPES})
  set(SWIG_COMMON_INCLUDES "${SWIG_COMMON_INCLUDES}%}\n")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/_smil.i
                 ${SWIG_MAIN_INTERFACE})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/_smilCommon.i
                 ${SWIG_COMMON_INTERFACE})

  set_source_files_properties(${SWIG_MAIN_INTERFACE} ${SWIG_COMMON_INTERFACE}
                              PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(${SWIG_MAIN_INTERFACE} ${SWIG_COMMON_INTERFACE}
                              PROPERTIES GENERATED TRUE)

  if(WRAP_PYTHON)
    # Trick to avoid to rebuild every time even without the dependencies having
    # changed
    set_source_files_properties(${SWIG_MAIN_INTERFACE}
                                PROPERTIES SWIG_MODULE_NAME smil_Python)
    message(STATUS "Wrap for python version ${Python3_VERSION}")
    if(MSVC)
      # Anaconda does not provide Python libraries with debug symbols
      if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_BUILD_TYPE
            "Release"
            CACHE STRING "" FORCE)
        message(STATUS "Changing build type to 'Release' for Python on Windows")
      endif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    endif(MSVC)

    set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilPython")
    swig_add_library(
      smil_Python
      LANGUAGE python
      SOURCES ${SWIG_MAIN_INTERFACE})
    target_link_libraries(smil_Python PRIVATE ${SMIL_LIBS} Python3::Python)
    file(
      WRITE ${LIBRARY_OUTPUT_PATH}/smilPython.pth
      "${SMIL_LIBRARIES_INSTALL_PATH}\n${SMIL_LIBRARIES_INSTALL_PATH}/smilPython"
    )
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/__init__.py
                   ${LIBRARY_OUTPUT_PATH}/smilPython/__init__.py COPYONLY)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/CMake/smilFuncRewrite.py
      ${LIBRARY_OUTPUT_PATH}/smilPython/smilFuncRewrite.py COPYONLY)
    # SET_TARGET_PROPERTIES(_smilPython PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    # ${LIBRARY_OUTPUT_PATH}/smilPython)
    install(TARGETS smil_Python
            LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}/smilPython
                    COMPONENT python)
    install(
      FILES ${LIBRARY_OUTPUT_PATH}/smilPython/smil_Python.py
            ${LIBRARY_OUTPUT_PATH}/smilPython/smilFuncRewrite.py
            ${LIBRARY_OUTPUT_PATH}/smilPython/__init__.py
      DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}/smilPython
      COMPONENT python)

    if(UNIX)
      if(CONDA_PREFIX)
        install(
          FILES ${LIBRARY_OUTPUT_PATH}/smilPython.pth
          DESTINATION
            ${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION}/site-packages
          COMPONENT python)
      else(CONDA_PREFIX)
        install(
          FILES ${LIBRARY_OUTPUT_PATH}/smilPython.pth
          DESTINATION
            ${CMAKE_INSTALL_PREFIX}/lib${LIBSUFFIX}/python${Python3_VERSION}/dist-packages
          COMPONENT python)
      endif(CONDA_PREFIX)
    endif(UNIX)

    add_dependencies(python smil_Python)
  endif(WRAP_PYTHON)

  if(WRAP_JAVA)
    set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}")
    swig_add_module(smilJava LANGUAGE java SOURCES ${SWIG_MAIN_INTERFACE})
    swig_link_libraries(smilJava ${SMIL_LIBS})
    install(TARGETS smilJava LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                                     COMPONENT java)
    install(
      DIRECTORY ${LIBRARY_OUTPUT_PATH}/smilJava
      DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}/smilJava
      COMPONENT java)
    add_dependencies(java smilJava)
  endif(WRAP_JAVA)
  #
  if(WRAP_OCTAVE)
    set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilOctave")
    swig_add_module(smilOctave LANGUAGE octave SOURCES ${SWIG_MAIN_INTERFACE})
    set_target_properties(smilOctave PROPERTIES PREFIX "" SUFFIX ".oct")
    swig_link_libraries(smilOctave ${SMIL_LIBS})
    install(TARGETS smilOctave
            LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH} COMPONENT octave)
    add_dependencies(octave smilOctave)
  endif(WRAP_OCTAVE)
  #
  if(WRAP_RUBY)
    set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilRuby")
    swig_add_module(smilRuby LANGUAGE ruby SOURCES ${SWIG_MAIN_INTERFACE})
    set_target_properties(smilRuby PROPERTIES PREFIX "")
    swig_link_libraries(smilRuby ${SMIL_LIBS})
    install(TARGETS smilRuby LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                                     COMPONENT ruby)
    add_dependencies(ruby smilRuby)
  endif(WRAP_RUBY)
  #
  if(WRAP_PHP)
    set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilPhp")
    include_directories(${LIBRARY_OUTPUT_PATH}/smilPhp)
    swig_add_module(smilPhp LANGUAGE php SOURCES ${SWIG_MAIN_INTERFACE})
    set_target_properties(smilPhp PROPERTIES PREFIX "")
    swig_link_libraries(smilPhp ${SMIL_LIBS})
    install(TARGETS smilPhp LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                                    COMPONENT php)
    add_dependencies(php smilPhp)
  endif(WRAP_PHP)
  #
endif(USE_WRAPPER)

add_smil_tests(Smil smilCore smilIO smilGui smilMorpho ${SMIL_EXT_DEPS})

include(Preinstall)

# INCLUDE(SmilInstallGen)

option(BUILD_PACKAGE "Build CPack packages" OFF)
mark_as_advanced(BUILD_PACKAGE)
if(BUILD_PACKAGE)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/CPack")
  include(SmilCPack)
endif(BUILD_PACKAGE)
