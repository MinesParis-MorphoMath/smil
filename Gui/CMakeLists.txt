# Smil
# Copyright (c) 2010 Matthieu Faessel
#
# This file is part of Smil.
#
# Smil is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Smil is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Smil.  If not, see
# <http://www.gnu.org/licenses/>.


IF(BUILD_GUI)
	SET(MODULE_NAME Gui)
	SET(MODULE_DEPS ${SMIL_LIB_PREFIX}Core)

	INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

	FILE(GLOB QT_UIS Qt/*.ui)
	FILE(GLOB QT_SRCS Qt/*.cpp)
	FILE(GLOB QT_HEADERS Qt/*.h)

	QT4_WRAP_UI(QT_UIS_H ${QT_UIS})
	QT4_WRAP_CPP(QT_MOC ${QT_HEADERS})

	FILE(GLOB QT_SRCS Qt/*.cpp)

	ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS} ADDITIONAL_SOURCES ${QT_SRCS} ${QT_UIS_H} ${QT_MOC})





	# ADD_LIBRARY(smilCoreGui ${SRCS} ${${MODULE_NAME}_UIS_H} ${${MODULE_NAME}_MOC})
	# TARGET_LINK_LIBRARIES(smilCoreGui ${QT_LIBRARIES})

ENDIF(BUILD_GUI)
