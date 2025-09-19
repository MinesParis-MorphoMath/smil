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

set(MODULE_NAME Gui)
set(MODULE_DEPS)
include_directories(
  ../Core/include ../Core/include/private ${CMAKE_CURRENT_BINARY_DIR}
  ../IO/include ../IO/include/private)

set(ADD_SRCS)

if(USE_QT)
  file(GLOB SRCS Qt/*.cpp)

  file(GLOB QT_UIS Qt/PureQt/*.ui)
  file(GLOB QT_SRCS Qt/PureQt/*.cpp)
  file(GLOB QT_HEADERS Qt/PureQt/*.h)

  if(USE_QWT)
    list(APPEND MODULE_DEPS ${QWT_LIBRARY})
  else(USE_QWT)
    file(GLOB QWT_FILES Qt/PureQt/PlotWidget.*)
    list(REMOVE_ITEM QT_SRCS ${QWT_FILES})
    list(REMOVE_ITEM QT_HEADERS ${QWT_FILES})
  endif(USE_QWT)

  if(USE_QT_VERSION EQUAL 5)
    qt5_wrap_ui(QT_UIS_H ${QT_UIS})
    qt5_wrap_cpp(QT_MOC ${QT_HEADERS})
  else() # QT4
    qt4_wrap_ui(QT_UIS_H ${QT_UIS})
    qt4_wrap_cpp(QT_MOC ${QT_HEADERS})
  endif()

  list(APPEND ADD_SRCS ${SRCS} ${QT_SRCS} ${QT_UIS_H} ${QT_MOC})

endif(USE_QT)

if(USE_AALIB)
  include_directories(AALib)
endif(USE_AALIB)

add_smil_library(${MODULE_NAME} ${MODULE_DEPS} ADDITIONAL_SOURCES ${ADD_SRCS})
add_smil_tests(${MODULE_NAME} ${MODULE_DEPS} ${SMIL_LIB_PREFIX}Core
               ${SMIL_LIB_PREFIX}IO)
