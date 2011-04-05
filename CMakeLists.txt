PROJECT(Smil)


CMAKE_MINIMUM_REQUIRED(VERSION 2.6)


LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/DCore/cmake")
INCLUDE(OptimizeForArchitecture)

###### COMPILATION PATHS ######

IF(NOT LIBRARY_OUTPUT_PATH)
	SET(LIBRARY_OUTPUT_PATH 
		${PROJECT_BINARY_DIR}/lib CACHE PATH 
		"Single output directory for building all libraries."
	)
	MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH)
ENDIF(NOT LIBRARY_OUTPUT_PATH)



IF(NOT EXECUTABLE_OUTPUT_PATH)
	SET(EXECUTABLE_OUTPUT_PATH 
		${PROJECT_BINARY_DIR}/bin CACHE PATH 
		"Single output directory for building all binaries."
	)
	MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH)
ENDIF(NOT EXECUTABLE_OUTPUT_PATH)

OPTION(USE_OPTIMIZATION "Use optimizations" ON)
IF(USE_OPTIMIZATION)
#    ADD_DEFINITIONS("-ftree-vectorize -ftree-vectorizer-verbose=1")
#     ADD_DEFINITIONS("-ftree-vectorize -g -ffast-math  -march=core2 -mtune=core2  -msse2 -fdump-tree-optimized")
	AddCompilerFlag("-ftree-vectorize")
	OptimizeForArchitecture()
	MESSAGE(STATUS "Compiler flags: "${CMAKE_CXX_FLAGS})
ENDIF(USE_OPTIMIZATION)


OPTION(BUILD_SHARED_LIBS "Build shared libs" ON)
IF(BUILD_SHARED_LIBS)
	ADD_DEFINITIONS("-DSHARED_LIBS")
ENDIF(BUILD_SHARED_LIBS)


OPTION(USE_OPEN_MP "Use OpenMP parallelization" OFF)
IF(USE_OPEN_MP)
    ADD_DEFINITIONS("-DMB_USE_OPEN_MP -fopenmp")
    LIST(APPEND MB_EXTERNAL_DEPS gomp pthread)
ENDIF(USE_OPEN_MP)


OPTION(WRAP_PYTHON "Wrap Python" FALSE)
OPTION(WRAP_JAVA "Wrap Java" FALSE)


ADD_DEFINITIONS("-fPIC")


ADD_SUBDIRECTORY(DCore)
