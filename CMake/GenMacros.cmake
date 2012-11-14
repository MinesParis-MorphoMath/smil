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


# ADD_SMIL_LIBRARY(libName libDeps [ADDITIONAL_SOURCES add_src1 ...] [EXCLUDED_SOURCES excl_src1 ...])

MACRO(ADD_SMIL_LIBRARY _LIB_NAME)

	PARSE_ARGUMENTS(_TARGET
		"ADDITIONAL_SOURCES;EXCLUDED_SOURCES"
		""
		${ARGN}
	)
	
	
	SET(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
	SET(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})
		
	FILE(GLOB LIB_SRCS src/*.cpp ${_TARGET_ADDITIONAL_SOURCES})

	IF(LIB_SRCS)
		ADD_LIBRARY(${LIB_NAME} ${LIB_SRCS} ${_TARGET_ADDITIONAL_SOURCES})
		IF(MINGW)
			SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES PREFIX "")
		ENDIF(MINGW)
		IF(LIB_DEPS)
			TARGET_LINK_LIBRARIES(${LIB_NAME} ${LIB_DEPS}) #gomp pthread)
		ENDIF(LIB_DEPS)
		SET(SMIL_LIBS ${SMIL_LIBS} ${LIB_NAME} PARENT_SCOPE)
	ENDIF(LIB_SRCS)


	IF(USE_WRAPPER AND SWIG_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)
	
		  SET(SWIG_INTERFACE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)
		  SET(SWIG_INTERFACE_FILES ${SWIG_INTERFACE_FILES} ${SWIG_INTERFACE_FILE} PARENT_SCOPE)

		  SET(CPP_LIB_NAME ${LIB_NAME}Cpp)
		  SET(PYTHON_LIB_NAME ${LIB_NAME}Python)
		  SET(JAVA_LIB_NAME ${LIB_NAME}Java)
		  SET(OCTAVE_LIB_NAME ${LIB_NAME}Octave)
		  SET(RUBY_LIB_NAME ${LIB_NAME}Ruby)

  		  SET_SOURCE_FILES_PROPERTIES(${LIB_NAME}.i PROPERTIES CPLUSPLUS ON)
		  
		  IF(WRAP_CPP)
			SMIL_WRAP_CPP(${LIB_NAME} ${CPP_LIB_NAME})
			TARGET_LINK_LIBRARIES(${CPP_LIB_NAME} ${LIB_DEPS})
		  ENDIF(WRAP_CPP)
# 		  
		  IF(WRAP_PYTHON)
			SWIG_ADD_MODULE(${PYTHON_LIB_NAME} python ${LIB_NAME}.i)
			SWIG_LINK_LIBRARIES(${PYTHON_LIB_NAME} ${LIB_DEPS} ${PYTHON_LIBRARIES} ${SWIG_DEPS})
			IF(LIB_SRCS)
			      SWIG_LINK_LIBRARIES(${PYTHON_LIB_NAME} ${LIB_NAME})
			ENDIF(LIB_SRCS)
		  ENDIF(WRAP_PYTHON)
# 		  
		  IF(WRAP_JAVA)
			SET(CMAKE_SWIG_OUTDIR "${PROJECT_BINARY_DIR}/java")
			SWIG_ADD_MODULE(${JAVA_LIB_NAME} java ${LIB_NAME}.i)
			SWIG_LINK_LIBRARIES(${JAVA_LIB_NAME} ${LIB_DEPS})
			IF(LIB_SRCS)
			      SWIG_LINK_LIBRARIES(${JAVA_LIB_NAME} ${LIB_NAME})
			ENDIF(LIB_SRCS)
		  ENDIF(WRAP_JAVA)

		  IF(WRAP_OCTAVE)
			SWIG_ADD_MODULE(${OCTAVE_LIB_NAME} octave ${LIB_NAME}.i)
			SET_TARGET_PROPERTIES(${OCTAVE_LIB_NAME} PROPERTIES PREFIX "" SUFFIX ".oct")
			SWIG_LINK_LIBRARIES(${OCTAVE_LIB_NAME} ${LIB_DEPS} smilCore)
			IF(LIB_SRCS)
			      SWIG_LINK_LIBRARIES(${OCTAVE_LIB_NAME} ${LIB_NAME})
			ENDIF(LIB_SRCS)
		  ENDIF(WRAP_OCTAVE)
# 		  
		  IF(WRAP_RUBY)
			SWIG_ADD_MODULE(${RUBY_LIB_NAME} ruby ${LIB_NAME}.i)
			SET_TARGET_PROPERTIES(${RUBY_LIB_NAME} PROPERTIES PREFIX "")
			SWIG_LINK_LIBRARIES(${RUBY_LIB_NAME} ${LIB_DEPS} smilCore)
			IF(LIB_SRCS)
			      SWIG_LINK_LIBRARIES(${RUBY_LIB_NAME} ${LIB_NAME})
			ENDIF(LIB_SRCS)
		  ENDIF(WRAP_RUBY)
# 		  


	ENDIF(USE_WRAPPER AND SWIG_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)
	
ENDMACRO(ADD_SMIL_LIBRARY _LIB_NAME _LIB_DEPS)




MACRO(ADD_SMIL_TESTS _LIB_NAME)
	
	PARSE_ARGUMENTS(_TARGET
		"ADDITIONAL_SOURCES;EXCLUDED_SOURCES"
		""
		${ARGN}
	)
	SET(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
	SET(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})

	IF(BUILD_TEST)
		FILE(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  test/*.cpp)

		IF(TEST_SRCS)
			FOREACH(_SRC ${TEST_SRCS})
				STRING(REPLACE ".cpp" "" TEST_EXE_NAME ${_SRC})
				STRING(REPLACE "test/" "" TEST_EXE_NAME ${TEST_EXE_NAME})
				ADD_EXECUTABLE(${TEST_EXE_NAME} ${_SRC})
				TARGET_LINK_LIBRARIES(${TEST_EXE_NAME} ${LIB_DEPS})
				IF(TARGET ${LIB_NAME})
					TARGET_LINK_LIBRARIES(${TEST_EXE_NAME} ${LIB_NAME})
				ENDIF(TARGET ${LIB_NAME})
				STRING(SUBSTRING ${TEST_EXE_NAME} 0 4 _EXE_PREFIX)
				IF(${_EXE_PREFIX} STREQUAL "test")
				    ADD_TEST("${_MOD}_${TEST_EXE_NAME}" ${EXECUTABLE_OUTPUT_PATH}/${TEST_EXE_NAME})
				ENDIF(${_EXE_PREFIX} STREQUAL "test")
			ENDFOREACH(_SRC ${TEST_SOURCE_FILES})
		ENDIF(TEST_SRCS)
	ENDIF(BUILD_TEST)
	
ENDMACRO(ADD_SMIL_TESTS _LIB_NAME)


##### LIST_CONTAINS #####

#  http://www.cmake.org/Wiki/CMakeMacroListOperations#LIST_CONTAINS

MACRO(LIST_CONTAINS var value)
  SET(${var})
  FOREACH (value2 ${ARGN})
    IF (${value} STREQUAL ${value2})
      SET(${var} TRUE)
    ENDIF (${value} STREQUAL ${value2})
  ENDFOREACH (value2)
ENDMACRO(LIST_CONTAINS)



##### PARSE_ARGUMENTS #####

# http://www.cmake.org/Wiki/CMakeMacroParseArguments

# There is a general convention for CMake commands that take optional flags and/or variable arguments. 
# Optional flags are all caps and are added to the arguments to turn on. 
# Variable arguments have an all caps identifier to determine where each variable argument list starts. 
# The PARSE_ARGUMENTS macro, defined below, can be used by other macros to parse arguments defined in this way. 
# Note that this macro relies on the  LIST_CONTAINS command. 

MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name})
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} FALSE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    LIST_CONTAINS(is_arg_name ${arg} ${arg_names})
    IF (is_arg_name)
      SET(${prefix}_${current_arg_name} ${current_arg_list})
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSE (is_arg_name)
      LIST_CONTAINS(is_option ${arg} ${option_names})
      IF (is_option)
	SET(${prefix}_${arg} TRUE)
      ELSE (is_option)
	SET(current_arg_list ${current_arg_list} ${arg})
      ENDIF (is_option)
    ENDIF (is_arg_name)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)


