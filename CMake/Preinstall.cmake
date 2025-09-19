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

file(READ LICENSE.txt LICENCE_TXT)
string(REGEX REPLACE "([^\n]*\n)" "// \\1" LICENCE_TXT_WITH_SLASH
                     "${LICENCE_TXT}")
string(REGEX REPLACE "([^\n]*\n)" "# \\1" LICENCE_TXT_WITH_SHARP
                     "${LICENCE_TXT}")

# Definitions
get_directory_property(_USE_SMIL_DEFINITIONS COMPILE_DEFINITIONS)
list(REMOVE_DUPLICATES _USE_SMIL_DEFINITIONS)
set(USE_SMIL_DEFINITIONS)
set(USE_SMIL_DEFINES)
foreach(DEF ${_USE_SMIL_DEFINITIONS})
  if(NOT ${DEF} MATCHES "\\$.*")
    list(APPEND USE_SMIL_DEFINITIONS "-D${DEF} ")
    list(APPEND USE_SMIL_DEFINES ${DEF})
  endif()
endforeach(DEF ${_USE_SMIL_DEFINITIONS})
if(USE_OPEN_MP)
  list(APPEND USE_SMIL_DEFINITIONS ${OpenMP_CXX_FLAGS})
endif(USE_OPEN_MP)

# Include directories
get_directory_property(USE_SMIL_INCLUDE_DIRS INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES USE_SMIL_INCLUDE_DIRS)
# Remove build includes
foreach(DIR ${USE_SMIL_INCLUDE_DIRS})
  string(REGEX MATCH "${PROJECT_SOURCE_DIR}.*|${PROJECT_BINARY_DIR}.*"
               IS_SMIL_DIR ${DIR})
  if(IS_SMIL_DIR)
    list(REMOVE_ITEM USE_SMIL_INCLUDE_DIRS ${DIR})
  endif()
endforeach()
set(USE_SMIL_INCLUDE_DIRS_LOCAL
    ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    ${USE_SMIL_INCLUDE_DIRS})
list(INSERT USE_SMIL_INCLUDE_DIRS 0 ${SMIL_HEADERS_INSTALL_PATH})

# Libraries
set(USE_SMIL_LIBS ${SMIL_LIBS})
set(USE_SMIL_LIB_DIRS)
foreach(LIB ${SMIL_EXT_DEPS})
  get_filename_component(LIB_NAME ${LIB} NAME_WE)
  get_filename_component(LIB_DIR ${LIB} PATH)
  if(NOT WIN32 OR NOT MSVC)
    string(REGEX REPLACE "^${CMAKE_FIND_LIBRARY_PREFIXES}" "" LIB_NAME
                         ${LIB_NAME}) # remove lib prefix to lib name
  endif(NOT WIN32 OR NOT MSVC)
  list(APPEND USE_SMIL_LIBS "${LIB_NAME}")
  list(APPEND USE_SMIL_LIB_DIRS "${LIB_DIR}")
endforeach()
# check for empty list: when no dependency is available, list is empty and
# remove_duplicates fails
if(USE_SMIL_LIB_DIRS)
  list(REMOVE_DUPLICATES USE_SMIL_LIB_DIRS)
endif(USE_SMIL_LIB_DIRS)
list(REMOVE_ITEM USE_SMIL_LIBS "debug" "optimized")

set(USE_SMIL_LIB_DIRS_LOCAL ${LIBRARY_OUTPUT_PATH} ${USE_SMIL_LIB_DIRS})
set(USE_SMIL_LIB_DIRS ${SMIL_LIBRARIES_INSTALL_PATH} ${USE_SMIL_LIB_DIRS})

# Generate main header Smil.h
set(MAIN_HEADER_STR "${LICENCE_TXT_WITH_SLASH}\n")
list(APPEND MAIN_HEADER_STR
     "#ifndef SMIL_GLOBAL_HEADER\n#define SMIL_GLOBAL_HEADER\n\n\n")

foreach(DEF ${USE_SMIL_DEFINES})
  string(REGEX REPLACE "=.*" "" EMPTY_DEF ${DEF})
  string(REPLACE "=" " " DEF ${DEF})
  list(APPEND MAIN_HEADER_STR "#ifndef ${EMPTY_DEF}\n#define ${DEF}\n#endif\n")
endforeach()

list(APPEND MAIN_HEADER_STR "\n")
foreach(HEADER ${SMIL_GLOBAL_HEADERS})
  list(APPEND MAIN_HEADER_STR "#include \"${HEADER}\"\n")
endforeach()
list(APPEND MAIN_HEADER_STR "\n#endif\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/Smil.h ${MAIN_HEADER_STR})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/Smil.h
  DESTINATION include/Smil
  COMPONENT dev)

# Generate cmake file UseSmil.cmake (local)

set(CMAKE_USE_SMIL_STR "${LICENCE_TXT_WITH_SHARP}\n")
string(REPLACE ";" " " BUF "${USE_SMIL_INCLUDE_DIRS_LOCAL}")
list(APPEND CMAKE_USE_SMIL_STR "INCLUDE_DIRECTORIES(${BUF})\n\n")
string(REPLACE ";" " " BUF "${USE_SMIL_LIB_DIRS_LOCAL}")
list(APPEND CMAKE_USE_SMIL_STR "LINK_DIRECTORIES(${BUF})\n\n")
list(APPEND CMAKE_USE_SMIL_STR "ADD_DEFINITIONS(${USE_SMIL_DEFINITIONS})\n\n")
string(REPLACE ";" " " BUF "${USE_SMIL_LIBS}")
list(APPEND CMAKE_USE_SMIL_STR "SET(SMIL_LIBRARIES ${BUF})")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/UseSmil.cmake ${CMAKE_USE_SMIL_STR})

# Generate cmake file UseSmil.cmake (for installation)

set(CMAKE_USE_SMIL_STR "${LICENCE_TXT_WITH_SHARP}\n")
string(REPLACE ";" " " BUF "${USE_SMIL_INCLUDE_DIRS}")
list(APPEND CMAKE_USE_SMIL_STR "INCLUDE_DIRECTORIES(${BUF})\n\n")
string(REPLACE ";" " " BUF "${USE_SMIL_LIB_DIRS}")
list(APPEND CMAKE_USE_SMIL_STR "LINK_DIRECTORIES(${BUF})\n\n")
list(APPEND CMAKE_USE_SMIL_STR "ADD_DEFINITIONS(${USE_SMIL_DEFINITIONS})\n\n")
string(REPLACE ";" " " BUF "${USE_SMIL_LIBS}")
list(APPEND CMAKE_USE_SMIL_STR "SET(SMIL_LIBRARIES ${BUF})")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/UseSmil.cmake.install
     ${CMAKE_USE_SMIL_STR})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/UseSmil.cmake.install
  DESTINATION include/Smil
  RENAME UseSmil.cmake
  COMPONENT dev)

# Generate Qt .pri file
set(QT_PRI_SMIL_STR "${LICENCE_TXT_WITH_SHARP}\n")
list(APPEND QT_PRI_SMIL_STR "# QT SUBPROJECT FILE\n")
list(
  APPEND
  QT_PRI_SMIL_STR
  "# To be included in the main .pro file:\n#include(/usr/local/include/Smil/Smil.pri)\n\n"
)
foreach(DIR ${USE_SMIL_INCLUDE_DIRS})
  list(APPEND QT_PRI_SMIL_STR "INCLUDEPATH +=  ${DIR}\n")
endforeach()
list(APPEND QT_PRI_SMIL_STR "\n")
foreach(DIR ${USE_SMIL_LIB_DIRS})
  list(APPEND QT_PRI_SMIL_STR "LIBS +=  -L${DIR}\n")
endforeach()
list(APPEND QT_PRI_SMIL_STR "\n")
foreach(LIB ${USE_SMIL_LIBS})
  list(APPEND QT_PRI_SMIL_STR "LIBS +=  -l${LIB}\n")
endforeach()
list(APPEND QT_PRI_SMIL_STR "\n")
list(APPEND QT_PRI_SMIL_STR "QMAKE_CXXFLAGS +=  ${USE_SMIL_DEFINITIONS}\n")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/Smil.pri ${QT_PRI_SMIL_STR})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/Smil.pri
  DESTINATION include/Smil
  COMPONENT dev)

# Generate pkg-config .pc file
list(APPEND PKG_CONFIG_SMIL_DEPS "-lstdc++" "-lm")
# Caution: library order is important for dependency resolution
list(APPEND PKG_CONFIG_SMIL_LIBS " -lsmilBase -lsmilCore -lsmilGui"
     " -lsmilIO -lsmilMorpho -lsmilRGB" " -lsmilAdvanced")
if(USE_OPEN_MP)
  list(APPEND PKG_CONFIG_SMIL_CFLAGS " -fopenmp")
  list(APPEND PKG_CONFIG_SMIL_LIBS " -fopenmp")
endif(USE_OPEN_MP)
if(USE_OPTIMIZATION)
  list(APPEND PKG_CONFIG_SMIL_CFLAGS "${CMAKE_CXX_FLAGS}")
endif(USE_OPTIMIZATION)
# static (.cpp) dependencies
if(USE_PNG)
  list(APPEND PKG_CONFIG_SMIL_DEPS ${PNG_LIBRARIES})
endif(USE_PNG)
if(USE_JPEG)
  list(APPEND PKG_CONFIG_SMIL_DEPS ${JPEG_LIBRARIES})
endif(USE_JPEG)
if(USE_TIFF)
  list(APPEND PKG_CONFIG_SMIL_DEPS ${TIFF_LIBRARIES})
endif(USE_TIFF)
if(USE_CURL)
  list(APPEND PKG_CONFIG_SMIL_DEPS ${CURL_LIBRARIES})
endif(USE_CURL)
# dynamic (.hpp/.hxx) dependencies
if(USE_FREETYPE)
  list(APPEND PKG_CONFIG_SMIL_LIBS " ${FREETYPE_LIBRARIES}")
  # Hopefully get Freetype include directory...
  list(APPEND PKG_CONFIG_SMIL_CFLAGS " -I${FREETYPE_INCLUDE_DIR_FTHEADER}")
endif(USE_FREETYPE)
if(USE_QWT)
  list(APPEND PKG_CONFIG_SMIL_LIBS " -lqwt")
  list(APPEND PKG_CONFIG_SMIL_CFLAGS " -I${QWT_INCLUDE_DIR}")
endif(USE_QWT)
if(USE_AALIB)
  list(APPEND PKG_CONFIG_SMIL_LIBS " -l${AALIB_LIBRARY}")
endif(USE_AALIB)
if(USE_QT)
  if(USE_QT_VERSION EQUAL 5)
    # Qt libs are needed by Smil header files...
    list(APPEND PKG_CONFIG_SMIL_LIBS " -lQt5Widgets -lQt5Gui -lQt5Core")
    list(APPEND PKG_CONFIG_SMIL_CFLAGS " -fPIC")
    foreach(INC ${Qt5Widgets_INCLUDE_DIRS})
      list(APPEND PKG_CONFIG_SMIL_CFLAGS " -I${INC}")
    endforeach()
  else()
    list(APPEND PKG_CONFIG_SMIL_LIBS " -lQtGui -lQtCore")
    foreach(INC ${QT_INCLUDES})
      list(APPEND PKG_CONFIG_SMIL_CFLAGS " -I${INC}")
    endforeach()
  endif(USE_QT_VERSION EQUAL 5)
endif(USE_QT)
# Generate spaces-separated strings
if(BUILD_SHARED_LIBS)
  # Smil libs are shared objects put Smil library dependencies into Libs.private
  # field
  foreach(LIB ${PKG_CONFIG_SMIL_DEPS})
    list(APPEND PKG_CONFIG_SMIL_LIBS_PRIV " ${LIB}")
  endforeach()
else()
  # Smil libs are archives applications have to link with Smil library
  # dependencies
  foreach(LIB ${PKG_CONFIG_SMIL_DEPS})
    list(APPEND PKG_CONFIG_SMIL_LIBS " ${LIB}")
  endforeach()
endif(BUILD_SHARED_LIBS)

list(
  APPEND
  PKG_CONFIG_SMIL_STR
  "prefix=${CMAKE_INSTALL_PREFIX}\n"
  "libdir=\${prefix}/lib${LIBSUFFIX}/Smil\n"
  "includedir=\${prefix}/include\n"
  "\n"
  "Name: Smil\n"
  "Description: Simple (but efficient) Morphological Image Library\n"
  "URL: https://smil.cmm.minesparis.psl.eu/doc/index.html\n"
  "Version: ${SMIL_VERSION}\n"
  "Libs: -L\${libdir} ${PKG_CONFIG_SMIL_LIBS}\n"
  "Libs.private: ${PKG_CONFIG_SMIL_LIBS_PRIV}\n"
  "Cflags: -I\${includedir}/Smil ${PKG_CONFIG_SMIL_CFLAGS}\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/smil.pc ${PKG_CONFIG_SMIL_STR})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/smil.pc
  DESTINATION lib${LIBSUFFIX}/pkgconfig
  COMPONENT dev)
