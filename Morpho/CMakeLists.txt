
INCLUDE_DIRECTORIES(. include include/private)

# FILE(GLOB SRCS src/*.cpp c/*.cpp)
# ADD_LIBRARY(mambaCore ${SRCS})
# TARGET_LINK_LIBRARIES(mambaCore ${MB_EXTERNAL_DEPS})

# FILE(GLOB SRCS c/*.cpp)
# ADD_LIBRARY(mambaCore_C ${SRCS})
# TARGET_LINK_LIBRARIES(mambaCore_C ${MB_EXTERNAL_DEPS})

SET(EXTERNAL_DEPS)


FILE(GLOB SRCS src/*.cpp)
# ADD_LIBRARY(smilMorpho ${SRCS})
# TARGET_LINK_LIBRARIES(smilMorpho ${EXTERNAL_DEPS}) #gomp pthread)

# TARGET_LINK_LIBRARIES(smilMorpho smilCore)



IF(WRAP_JAVA)
ENDIF(WRAP_JAVA)


IF(WRAP_PYTHON)
      IF(SWIG_FOUND AND PYTHONLIBS_FOUND)
	  SET_SOURCE_FILES_PROPERTIES(smilMorphoPython.i PROPERTIES CPLUSPLUS ON)
 	  
	  SWIG_ADD_MODULE(smilMorphoPython python smilMorphoPython.i)
	  
# 	  SWIG_ADD_MODULE(smilCoreJava java smilCore.i)
 	  SWIG_LINK_LIBRARIES(smilMorphoPython smilCore ${PYTHON_LIBRARIES})
#  	  SWIG_LINK_LIBRARIES(smilMorphoPython smilMorpho smilCore ${PYTHON_LIBRARIES})
# 	  SWIG_LINK_LIBRARIES(smilCoreJava smilCore)
	  
 	  SET_TARGET_PROPERTIES(_smilMorphoPython PROPERTIES LINKER_LANGUAGE CXX)
#  	  SET_TARGET_PROPERTIES(smilCoreJava PROPERTIES LINKER_LANGUAGE CXX)

      ENDIF(SWIG_FOUND AND PYTHONLIBS_FOUND)


ENDIF(WRAP_PYTHON)
