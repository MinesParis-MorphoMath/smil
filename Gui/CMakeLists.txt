# Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
#     * Neither the name of the University of California, Berkeley nor the
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


IF(BUILD_GUI)
	SET(MODULE_NAME Gui)
	SET(MODULE_DEPS ${SMIL_LIB_PREFIX}Core)

	INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

	FILE(GLOB QT_UIS Qt/*.ui)
	FILE(GLOB QT_SRCS Qt/*.cpp)
	FILE(GLOB QT_HEADERS Qt/*.h)

	QT4_WRAP_UI(QT_UIS_H ${QT_UIS})
	QT4_WRAP_CPP(QT_MOC ${QT_HEADERS})


	ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS} ADDITIONAL_SOURCES ${QT_SRCS} ${QT_UIS_H} ${QT_MOC})





	# ADD_LIBRARY(smilCoreGui ${SRCS} ${${MODULE_NAME}_UIS_H} ${${MODULE_NAME}_MOC})
	# TARGET_LINK_LIBRARIES(smilCoreGui ${QT_LIBRARIES})

ENDIF(BUILD_GUI)
