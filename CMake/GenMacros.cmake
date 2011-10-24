
# ADD_SMIL_LIBRARY(libName libDeps [ADDITIONAL_SOURCES add_src1 ...] [EXCLUDED_SOURCES excl_src1 ...])

MACRO(ADD_SMIL_LIBRARY _LIB_NAME)

	PARSE_ARGUMENTS(_TARGET
		"ADDITIONAL_SOURCES;EXCLUDED_SOURCES"
		""
		${ARGN}
	)
	SET(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
	SET(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})
		
	FILE(GLOB LIB_SRCS src/*.cpp)

	IF(LIB_SRCS)
		ADD_LIBRARY(${LIB_NAME} ${LIB_SRCS} ${_TARGET_ADDITIONAL_SOURCES})
		IF(LIB_DEPS)
			TARGET_LINK_LIBRARIES(${LIB_NAME} ${LIB_DEPS}) #gomp pthread)
		ENDIF(LIB_DEPS)
		LIST(APPEND LIB_DEPS ${LIB_NAME})
		SET(SMIL_LIBS ${SMIL_LIBS} ${LIB_NAME})
		SET(SMIL_LIBS ${SMIL_LIBS} ${LIB_NAME} PARENT_SCOPE)
	ENDIF(LIB_SRCS)


	IF(WRAP_PYTHON)
	      IF(SWIG_FOUND AND PYTHONLIBS_FOUND)
		  SET(PYTHON_LIB_NAME ${LIB_NAME}Python)
		  SET(SWIG_INTERFACE_FILES ${SWIG_INTERFACE_FILES} ${PYTHON_LIB_NAME}.i PARENT_SCOPE)
		  SET_SOURCE_FILES_PROPERTIES(${PYTHON_LIB_NAME}.i PROPERTIES CPLUSPLUS ON)
		  
# 		  IF(USE_QT)
# 		      SET(CMAKE_SWIG_FLAGS -DUSE_QT)
# 		  ENDIF(USE_QT)

# 		  SWIG_ADD_MODULE(smilPython python ${PYTHON_LIB_NAME}.i)
# 		  
# 	# 	  SWIG_ADD_MODULE(smilCoreJava java smilCore.i)
# 		  SWIG_LINK_LIBRARIES(smilPython ${LIB_DEPS} ${PYTHON_LIBRARIES})
# 		  IF(USE_QT)
# 	# 	      SWIG_LINK_LIBRARIES(smilCorePython smilCoreGui)
# 		  ENDIF(USE_QT)
# 	# 	  SWIG_LINK_LIBRARIES(smilCoreJava smilCore)
# 		  
# 		  SET_TARGET_PROPERTIES(_smilPython PROPERTIES LINKER_LANGUAGE CXX)
	#  	  SET_TARGET_PROPERTIES(smilCoreJava PROPERTIES LINKER_LANGUAGE CXX)

	      ENDIF(SWIG_FOUND AND PYTHONLIBS_FOUND)


	ENDIF(WRAP_PYTHON)
	
ENDMACRO(ADD_SMIL_LIBRARY _LIB_NAME _LIB_DEPS)


MACRO(ADD_SMIL_TESTS _LIB_NAME _LIB_DEPS)
	
	PARSE_ARGUMENTS(_TARGET
		"ADDITIONAL_SOURCES;EXCLUDED_SOURCES"
		""
		${ARGN}
	)
	SET(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
	SET(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})
	LIST_CONTAINS(SELF_LIB_EXISTS ${LIB_NAME} ${SMIL_LIBS})
	IF(SELF_LIB_EXISTS)
		LIST(APPEND LIB_DEPS ${LIB_NAME})
	ENDIF(SELF_LIB_EXISTS)

	IF(BUILD_TEST)
		FILE(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  test/*.cpp)
		
		IF(TEST_SRCS)
			FOREACH(_SRC ${TEST_SRCS})
				STRING(REPLACE ".cpp" "" TEST_EXE_NAME ${_SRC})
				STRING(REPLACE "test/" "" TEST_EXE_NAME ${TEST_EXE_NAME})
				ADD_EXECUTABLE(${TEST_EXE_NAME} ${_SRC})
				TARGET_LINK_LIBRARIES(${TEST_EXE_NAME} ${LIB_DEPS})
				ADD_TEST("${_MOD}_${TEST_EXE_NAME}" ${EXECUTABLE_OUTPUT_PATH}/${TEST_EXE_NAME})
			ENDFOREACH(_SRC ${TEST_SOURCE_FILES})
		ENDIF(TEST_SRCS)
	ENDIF(BUILD_TEST)
	
ENDMACRO(ADD_SMIL_TESTS _LIB_NAME _LIB_DEPS)


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


