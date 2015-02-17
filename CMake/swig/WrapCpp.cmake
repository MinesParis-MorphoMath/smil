# Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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

MACRO(SMIL_WRAP_CPP LIB_NAME CPP_LIB_NAME)
	SET(CPP_WRAP_HEADERS_DIR ${PROJECT_BINARY_DIR}/include)
	FILE(MAKE_DIRECTORY ${CPP_WRAP_HEADERS_DIR})
	SET(XML_WRAP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}_wrap.xml)
	SET(CPP_WRAP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}CPP_wrap.cpp)
	SET(CPP_WRAP_HEADER_FILE ${CPP_WRAP_HEADERS_DIR}/${LIB_NAME}.h)
	GET_DIRECTORY_PROPERTY(CMAKE_INCLUDE_DIRS INCLUDE_DIRECTORIES)
	FOREACH(_DIR ${CMAKE_INCLUDE_DIRS})
	  SET(SWIG_INCLUDE_DIRS ${SWIG_INCLUDE_DIRS} "-I${_DIR}")
	ENDFOREACH(_DIR ${CMAKE_INCLUDE_DIRS})

	# Parse module files to detemine depends
	FILE(GLOB_RECURSE MODULE_HEADERS *.h *.hpp *.hxx)
	
	ADD_CUSTOM_COMMAND(
	    OUTPUT  ${XML_WRAP_FILE}
	    COMMAND ${SWIG_EXECUTABLE} -xml -c++ -xmllite
		${CMAKE_SWIG_FLAGS} 
		${SWIG_INCLUDE_DIRS} 
		-o ${XML_WRAP_FILE}
		${SWIG_INTERFACE_FILE}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	    COMMENT "Creating ${XML_WRAP_FILE} file"
	    DEPENDS ${LIB_NAME}.i ${MODULE_HEADERS}
	)
	
	SET(PY_XML_XPP_SCRIPT ${PROJECT_SOURCE_DIR}/CMake/swig/swigxmlcpp.py)
	ADD_CUSTOM_COMMAND(
	    OUTPUT  ${CPP_WRAP_FILE}
	    COMMAND ${PYTHON_EXECUTABLE} ${PY_XML_XPP_SCRIPT}
		${LIB_NAME} 
		"${XML_WRAP_FILE}"
		"${CPP_WRAP_FILE}"
		"${CPP_WRAP_HEADER_FILE}"
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	    COMMENT "Creating ${CPP_WRAP_FILE} file"
	    DEPENDS ${XML_WRAP_FILE} ${PY_XML_XPP_SCRIPT}
	)
	ADD_LIBRARY(${CPP_LIB_NAME} ${CPP_WRAP_FILE})
ENDMACRO(SMIL_WRAP_CPP)
