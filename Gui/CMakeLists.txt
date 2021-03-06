# Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SET(MODULE_NAME Gui)
SET(MODULE_DEPS)
INCLUDE_DIRECTORIES(../Core/include ../Core/include/private ${CMAKE_CURRENT_BINARY_DIR} ../IO/include ../IO/include/private)

SET(ADD_SRCS)

IF(USE_QT)
	FILE(GLOB SRCS Qt/*.cpp)
	
	FILE(GLOB QT_UIS Qt/PureQt/*.ui)
	FILE(GLOB QT_SRCS Qt/PureQt/*.cpp)
	FILE(GLOB QT_HEADERS Qt/PureQt/*.h)

	IF(USE_QWT)
	      LIST(APPEND MODULE_DEPS ${QWT_LIBRARY})
	ELSE(USE_QWT)
	      FILE(GLOB QWT_FILES Qt/PureQt/PlotWidget.*)
	      LIST(REMOVE_ITEM QT_SRCS ${QWT_FILES})
	      LIST(REMOVE_ITEM QT_HEADERS ${QWT_FILES})
	ENDIF(USE_QWT)

	IF(USE_QT_VERSION EQUAL 5)
            QT5_WRAP_UI(QT_UIS_H ${QT_UIS})
            QT5_WRAP_CPP(QT_MOC ${QT_HEADERS})
        ELSE() # QT4
            QT4_WRAP_UI(QT_UIS_H ${QT_UIS})
            QT4_WRAP_CPP(QT_MOC ${QT_HEADERS})
        ENDIF()
	

	LIST(APPEND ADD_SRCS ${SRCS} ${QT_SRCS} ${QT_UIS_H} ${QT_MOC})

ENDIF(USE_QT)

IF(USE_AALIB)
	INCLUDE_DIRECTORIES(AALib)
ENDIF(USE_AALIB)

ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS} ADDITIONAL_SOURCES ${ADD_SRCS})
ADD_SMIL_TESTS(${MODULE_NAME} ${MODULE_DEPS} ${SMIL_LIB_PREFIX}Core ${SMIL_LIB_PREFIX}IO)
