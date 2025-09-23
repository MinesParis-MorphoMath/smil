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

macro(INSTALL_LIB _LIB_NAME)
  install(
    TARGETS ${_LIB_NAME}
    LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH} COMPONENT library
    ARCHIVE DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH} COMPONENT library
    RUNTIME DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH} COMPONENT library)
endmacro(INSTALL_LIB _LIB_NAME)

# ADD_SMIL_LIBRARY(libName libDeps [ADDITIONAL_SOURCES add_src1 ...]
# [EXCLUDED_SOURCES excl_src1 ...])

macro(ADD_SMIL_LIBRARY _LIB_NAME)
  parse_arguments(
    _TARGET "ADDITIONAL_SOURCES;EXCLUDED_SOURCES;COMPILE_FLAGS;LINK_FLAGS" ""
    ${ARGN})

  string(TOUPPER ${_LIB_NAME} UPPER_LIB_NAME)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                      ${CMAKE_CURRENT_SOURCE_DIR}/include/private)
  string(REPLACE "${CMAKE_SOURCE_DIR}/" "" CURRENT_RELATIVE_SOURCE_DIR
                 ${CMAKE_CURRENT_SOURCE_DIR})
  string(REGEX MATCH "NSTypes/.*" IS_NSTYPE ${CURRENT_RELATIVE_SOURCE_DIR})

  list_contains(IS_BASE_MODULE ${_LIB_NAME} ${MODULES})
  if(IS_BASE_MODULE OR IS_NSTYPE)
    set(COMPONENT_PREFIX)
    set(COMPONENT_PREFIX_)
  else()
    set(COMPONENT_PREFIX ${_LIB_NAME})
    set(COMPONENT_PREFIX_ ${_LIB_NAME}-)
  endif()
  set(COMPONENT_LIST)

  set(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
  set(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})
  # Add non-smil libs to ext-deps
  if(_TARGET_DEFAULT_ARGS)
    string(REGEX REPLACE "smil[^;]*" "" LIB_EXT_DEPS ${_TARGET_DEFAULT_ARGS})
  endif()
  list(APPEND SMIL_EXT_DEPS ${LIB_EXT_DEPS})
  set(SMIL_EXT_DEPS
      ${SMIL_EXT_DEPS}
      PARENT_SCOPE)

  file(
    GLOB_RECURSE LIB_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.h *.hpp *.hxx)
  foreach(FCH ${LIB_HEADERS})
    install(
      FILES ${FCH}
      DESTINATION include/Smil/${CURRENT_RELATIVE_SOURCE_DIR}/
      RENAME ${FCH}
      COMPONENT ${COMPONENT_PREFIX_}dev)
  endforeach()
  list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}dev)

  if(IS_BASE_MODULE AND EXISTS
                        ${CMAKE_CURRENT_SOURCE_DIR}/include/D${MODULE_NAME}.h)
    set(SMIL_GLOBAL_HEADERS
        ${SMIL_GLOBAL_HEADERS}
        ${CURRENT_RELATIVE_SOURCE_DIR}/include/D${MODULE_NAME}.h
        PARENT_SCOPE)
  endif()

  file(GLOB LIB_SRCS src/*.cpp ${_TARGET_ADDITIONAL_SOURCES})
  if(_TARGET_EXCLUDED_SOURCES AND LIB_SRCS)
    file(GLOB_RECURSE EXCL_SRCS ${_TARGET_EXCLUDED_SOURCES})
    if(EXCL_SRCS)
      list(REMOVE_ITEM LIB_SRCS ${EXCL_SRCS})
    endif(EXCL_SRCS)
  endif(_TARGET_EXCLUDED_SOURCES AND LIB_SRCS)

  if(LIB_SRCS)
    add_library(${LIB_NAME} ${LIB_SRCS} ${_TARGET_ADDITIONAL_SOURCES})

    if(NOT MSVC)
      target_compile_options(
        ${LIB_NAME} PRIVATE -Wall -Wextra -Wsign-compare -Wno-strict-aliasing
                            -Wno-unused-function)
    endif()

    if(_TARGET_COMPILE_FLAGS)
      set_target_properties(${LIB_NAME} PROPERTIES COMPILE_FLAGS
                                                   ${_TARGET_COMPILE_FLAGS})
    endif(_TARGET_COMPILE_FLAGS)
    if(_TARGET_LINK_FLAGS)
      set_target_properties(${LIB_NAME} PROPERTIES LINK_FLAGS
                                                   ${_TARGET_LINK_FLAGS})
    endif(_TARGET_LINK_FLAGS)
    if(MINGW)
      set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")
      if(NOT BUILD_SHARED_LIBS)
        set_target_properties(${LIB_NAME} PROPERTIES SUFFIX ".dll.a")
      endif(NOT BUILD_SHARED_LIBS)
    endif(MINGW)
    if(LIB_DEPS)
      target_link_libraries(${LIB_NAME} ${LIB_DEPS}) # gomp pthread)
    endif(LIB_DEPS)
    set(SMIL_LIBS
        ${SMIL_LIBS} ${LIB_NAME}
        PARENT_SCOPE)

    set_target_properties(${LIB_NAME} PROPERTIES VERSION ${SMIL_VERSION})

    if(BUILD_SHARED_LIBS)
      install(
        TARGETS ${LIB_NAME}
        LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                COMPONENT ${COMPONENT_PREFIX_}base
        ARCHIVE DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                COMPONENT ${COMPONENT_PREFIX_}base
        RUNTIME DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                COMPONENT ${COMPONENT_PREFIX_}base)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}base)
    endif(BUILD_SHARED_LIBS)

    install(
      TARGETS ${LIB_NAME}
      LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
              COMPONENT ${COMPONENT_PREFIX_}dev
      ARCHIVE DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
              COMPONENT ${COMPONENT_PREFIX_}dev
      RUNTIME DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
              COMPONENT ${COMPONENT_PREFIX_}dev)
  endif(LIB_SRCS)

  if(USE_WRAPPER
     AND SWIG_FOUND
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)

    set(SWIG_INTERFACE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)
    set(SWIG_INTERFACE_FILES
        ${SWIG_INTERFACE_FILES} ${SWIG_INTERFACE_FILE}
        PARENT_SCOPE)

    set(CPP_LIB_NAME ${LIB_NAME}Cpp)
    set(PYTHON_LIB_NAME ${LIB_NAME}Python)
    set(JAVA_LIB_NAME ${LIB_NAME}Java)
    set(OCTAVE_LIB_NAME ${LIB_NAME}Octave)
    set(RUBY_LIB_NAME ${LIB_NAME}Ruby)
    set(PHP_LIB_NAME ${LIB_NAME}Php)

    set_source_files_properties(${LIB_NAME}.i PROPERTIES CPLUSPLUS ON)

    add_definitions(-DSWIG_WRAP_${UPPER_LIB_NAME})

    if(MSVC)
      add_definitions("/bigobj")
    endif(MSVC)

    if(WRAP_CPP)
      smil_wrap_cpp(${LIB_NAME} ${CPP_LIB_NAME})
      target_link_libraries(${CPP_LIB_NAME} ${LIB_DEPS})
      install(
        TARGETS ${CPP_LIB_NAME}
        LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                COMPONENT ${COMPONENT_PREFIX_}cpp
        ARCHIVE DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                COMPONENT ${COMPONENT_PREFIX_}cpp)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}cpp)
    endif(WRAP_CPP)
    #
    if(WRAP_PYTHON)
      # Trick to avoid to rebuild every time even without the dependencies
      # having changed
      set_source_files_properties(${LIB_NAME}.i PROPERTIES SWIG_MODULE_NAME
                                                           ${LIB_NAME}Python)

      set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilPython")
      swig_add_library(
        ${PYTHON_LIB_NAME}
        LANGUAGE python
        SOURCES ${LIB_NAME}.i)
      swig_link_libraries(${PYTHON_LIB_NAME} ${LIB_DEPS} ${PYTHON_LIBRARIES}
                          ${SWIG_DEPS})
      # SET_TARGET_PROPERTIES(_${PYTHON_LIB_NAME} PROPERTIES
      # LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/smilPython)
      if(LIB_SRCS)
        swig_link_libraries(${PYTHON_LIB_NAME} ${LIB_NAME} smilCore)
      endif(LIB_SRCS)
      install(TARGETS ${PYTHON_LIB_NAME}
              LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                      COMPONENT ${COMPONENT_PREFIX_}python)
      install(
        FILES ${LIBRARY_OUTPUT_PATH}/smilPython/${LIB_NAME}Python.py
        DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}/smilPython
        COMPONENT ${COMPONENT_PREFIX_}python)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}python)
      add_dependencies(python ${PYTHON_LIB_NAME})
    endif(WRAP_PYTHON)
    #
    if(WRAP_OCTAVE)
      set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilOctave")
      add_swig_module(${OCTAVE_LIB_NAME} octave ${LIB_NAME}.i)
      set_target_properties(${OCTAVE_LIB_NAME} PROPERTIES PREFIX "" SUFFIX
                                                                    ".oct")
      swig_link_libraries(${OCTAVE_LIB_NAME} ${LIB_DEPS} smilCore)
      if(LIB_SRCS)
        swig_link_libraries(${OCTAVE_LIB_NAME} ${LIB_NAME})
      endif(LIB_SRCS)
      install(TARGETS ${OCTAVE_LIB_NAME}
              LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                      COMPONENT ${COMPONENT_PREFIX_}octave)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}octave)
      add_dependencies(octave ${OCTAVE_LIB_NAME})
    endif(WRAP_OCTAVE)
    #
    if(WRAP_RUBY)
      set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilRuby")
      add_swig_module(${RUBY_LIB_NAME} ruby ${LIB_NAME}.i)
      set_target_properties(${RUBY_LIB_NAME} PROPERTIES PREFIX "")
      swig_link_libraries(${RUBY_LIB_NAME} ${LIB_DEPS} smilCore)
      if(LIB_SRCS)
        swig_link_libraries(${RUBY_LIB_NAME} ${LIB_NAME})
      endif(LIB_SRCS)
      install(TARGETS ${RUBY_LIB_NAME}
              LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                      COMPONENT ${COMPONENT_PREFIX_}ruby)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}ruby)
      add_dependencies(ruby ${RUBY_LIB_NAME})
    endif(WRAP_RUBY)
    #
    if(WRAP_PHP)
      set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilPhp")
      include_directories(${LIBRARY_OUTPUT_PATH}/smilPhp)
      add_swig_module(${PHP_LIB_NAME} php ${LIB_NAME}.i)
      set_target_properties(${PHP_LIB_NAME} PROPERTIES PREFIX "")
      swig_link_libraries(${PHP_LIB_NAME} ${LIB_DEPS} smilCore)
      if(LIB_SRCS)
        swig_link_libraries(${PHP_LIB_NAME} ${LIB_NAME})
      endif(LIB_SRCS)
      install(TARGETS ${PHP_LIB_NAME}
              LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                      COMPONENT ${COMPONENT_PREFIX_}php)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}php)
      add_dependencies(php ${PHP_LIB_NAME})
    endif(WRAP_PHP)

    # Keep Java in last position because of the "package" flags (which should
    # not have a general impact)
    if(WRAP_JAVA)
      set(CMAKE_SWIG_OUTDIR "${LIBRARY_OUTPUT_PATH}/smilJava")
      set_property(SOURCE ${LIB_NAME}.i PROPERTY SWIG_FLAGS -package
                                                 smil${COMPONENT_PREFIX}Java)
      add_swig_module(${JAVA_LIB_NAME} java ${LIB_NAME}.i)
      swig_link_libraries(${JAVA_LIB_NAME} ${LIB_DEPS})
      if(LIB_SRCS)
        swig_link_libraries(${JAVA_LIB_NAME} ${LIB_NAME} smilCore)
      endif(LIB_SRCS)
      install(TARGETS ${JAVA_LIB_NAME}
              LIBRARY DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
                      COMPONENT ${COMPONENT_PREFIX_}java)
      install(
        DIRECTORY ${LIBRARY_OUTPUT_PATH}/smil${COMPONENT_PREFIX}Java
        DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
        COMPONENT ${COMPONENT_PREFIX_}java)
      list(APPEND COMPONENT_LIST ${COMPONENT_PREFIX_}java)
      add_dependencies(java ${JAVA_LIB_NAME})
    endif(WRAP_JAVA)

  endif(
    USE_WRAPPER
    AND SWIG_FOUND
    AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}.i)

  list_contains(IS_IN_LIST ${COMPONENT_LIST} ${SMIL_INSTALL_COMPONENT_LIST};"")
  if(NOT IS_IN_LIST)
    set(SMIL_INSTALL_COMPONENT_LIST
        ${SMIL_INSTALL_COMPONENT_LIST} ${COMPONENT_LIST}
        PARENT_SCOPE)
  endif(NOT IS_IN_LIST)

endmacro(
  ADD_SMIL_LIBRARY
  _LIB_NAME
  _LIB_DEPS)

macro(ADD_SMIL_TESTS _LIB_NAME)

  parse_arguments(_TARGET "ADDITIONAL_SOURCES;EXCLUDED_SOURCES" "" ${ARGN})
  set(LIB_NAME ${SMIL_LIB_PREFIX}${_LIB_NAME})
  set(LIB_DEPS ${_TARGET_DEFAULT_ARGS} ${SMIL_EXT_DEPS})

  if(BUILD_TEST)
    set(FILE_PATTERNS test/test*.cpp test/bench*.cpp)
    if(WRAP_PYTHON)
      list(APPEND FILE_PATTERNS test/python*.cpp)
    endif(WRAP_PYTHON)
    file(
      GLOB TEST_SRCS
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      ${FILE_PATTERNS})

    if(_TARGET_EXCLUDED_SOURCES AND TEST_SRCS)
      file(
        GLOB_RECURSE EXCL_SRCS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        test/${_TARGET_EXCLUDED_SOURCES})
      if(EXCL_SRCS)
        list(REMOVE_ITEM TEST_SRCS ${EXCL_SRCS})
      endif(EXCL_SRCS)
    endif(_TARGET_EXCLUDED_SOURCES AND TEST_SRCS)

    string(REPLACE "${PROJECT_SOURCE_DIR}" "" MOD_NAME
                   ${CMAKE_CURRENT_SOURCE_DIR})
    if(MOD_NAME)
      string(REPLACE "/" "" MOD_NAME ${MOD_NAME})
      set(MOD_NAME ${MOD_NAME}_)
    endif(MOD_NAME)

    if(TEST_SRCS)
      foreach(_SRC ${TEST_SRCS})
        string(REPLACE ".cpp" "" TEST_NAME ${_SRC})
        string(REPLACE "test/" "" TEST_NAME ${TEST_NAME})
        string(REGEX REPLACE "(^[^_]+)_.+" "\\1" _EXE_PREFIX ${TEST_NAME})

        add_executable(${TEST_NAME} ${_SRC})
        if(TARGET ${LIB_NAME})
          target_link_libraries(${TEST_NAME} ${LIB_NAME} smilIO)
        endif(TARGET ${LIB_NAME})
        target_link_libraries(${TEST_NAME} ${LIB_DEPS})

        if(${_EXE_PREFIX} STREQUAL "test")
          add_test("${MOD_NAME}${TEST_NAME}"
                   ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME})
          add_dependencies(tests ${TEST_NAME})
        elseif(${_EXE_PREFIX} STREQUAL "python")
          add_test("${MOD_NAME}${TEST_NAME}"
                   ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME})
          target_link_libraries(${TEST_NAME} ${PYTHON_LIBRARIES})
          add_dependencies(tests ${TEST_NAME})
        elseif(${_EXE_PREFIX} STREQUAL "bench")
          add_test(NAME "${MOD_NAME}${TEST_NAME}"
                   COMMAND ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME})
          add_dependencies(benchs ${TEST_NAME})
        endif(${_EXE_PREFIX} STREQUAL "test")
      endforeach(_SRC ${TEST_SOURCE_FILES})
      # INSTALL(TARGETS ${RUBY_LIB_NAME} RUNTIME DESTINATION
      # ${SMIL_BINARIES_INSTALL_PATH} COMPONENT tests)
    endif(TEST_SRCS)
  endif(BUILD_TEST)

endmacro(ADD_SMIL_TESTS _LIB_NAME)

macro(FIND_SMIL_MODULE_SUBDIRECTORIES)
  file(
    GLOB _MOD_CMLISTS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/"
    "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
  # Filter out SampleModule
  list(REMOVE_ITEM _MOD_CMLISTS "SampleModule/CMakeLists.txt")
  foreach(_FCH ${_MOD_CMLISTS})
    get_filename_component(_DIR ${_FCH} PATH)
    string(TOUPPER ${_DIR} _USER_MOD)
    list(APPEND AVAILABLE_USER_MODS ${_DIR})
    option(ADD_USER_MOD_${_USER_MOD} "Build additional user module ${_DIR}" OFF)
    mark_as_advanced(ADDON_${_USER_MOD})
    if(ADD_USER_MOD_${_USER_MOD})
      include_directories(${_DIR} ${_DIR}/include ${_DIR}/include/private)
      add_subdirectory(${_DIR})
    endif(ADD_USER_MOD_${_USER_MOD})
  endforeach(_FCH ${_MOD_CMLISTS})
endmacro(FIND_SMIL_MODULE_SUBDIRECTORIES _DIR_NAME)

# LIST_CONTAINS #####

# http://www.cmake.org/Wiki/CMakeMacroListOperations#LIST_CONTAINS

macro(LIST_CONTAINS var value)
  set(${var})
  foreach(value2 ${ARGN})
    if(${value} STREQUAL ${value2})
      set(${var} TRUE)
    endif(${value} STREQUAL ${value2})
  endforeach(value2)
endmacro(LIST_CONTAINS)

# PARSE_ARGUMENTS #####

# http://www.cmake.org/Wiki/CMakeMacroParseArguments

# There is a general convention for CMake commands that take optional flags
# and/or variable arguments. Optional flags are all caps and are added to the
# arguments to turn on. Variable arguments have an all caps identifier to
# determine where each variable argument list starts. The PARSE_ARGUMENTS macro,
# defined below, can be used by other macros to parse arguments defined in this
# way. Note that this macro relies on the  LIST_CONTAINS command.

macro(PARSE_ARGUMENTS prefix arg_names option_names)
  set(DEFAULT_ARGS)
  foreach(arg_name ${arg_names})
    set(${prefix}_${arg_name})
  endforeach(arg_name)
  foreach(option ${option_names})
    set(${prefix}_${option} FALSE)
  endforeach(option)

  set(current_arg_name DEFAULT_ARGS)
  set(current_arg_list)
  foreach(arg ${ARGN})
    list_contains(is_arg_name ${arg} ${arg_names})
    if(is_arg_name)
      set(${prefix}_${current_arg_name} ${current_arg_list})
      set(current_arg_name ${arg})
      set(current_arg_list)
    else(is_arg_name)
      list_contains(is_option ${arg} ${option_names})
      if(is_option)
        set(${prefix}_${arg} TRUE)
      else(is_option)
        set(current_arg_list ${current_arg_list} ${arg})
      endif(is_option)
    endif(is_arg_name)
  endforeach(arg)
  set(${prefix}_${current_arg_name} ${current_arg_list})
endmacro(PARSE_ARGUMENTS)

# PKG-CONFIG #### Input: LIB_NAME (which is supposed to have an entry in
# pkg-config) [ STATIC ] Outputs: * LIB_NAME_PKG_FOUND: true if the pkg has been
# found * LIB_NAME_DEFS: compiler flags * LIB_NAME_LINK_DIRS: linker directories
# * LIB_NAME_LINK_LIBS: lib dependancies

macro(ADD_PKG_CONFIG_DEFS _LIB_NAME)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(${_LIB_NAME}_MOD ${_LIB_NAME})

    if(${_LIB_NAME}_MOD_FOUND)
      set(${_LIB_NAME}_PKG_FOUND TRUE)
      set(${_LIB_NAME}_DEFS ${${_LIB_NAME}_MOD_CFLAGS})
      set(${_LIB_NAME}_LINK_DIRS)
      set(${_LIB_NAME}_LINK_LIBS)
      if("${ARGV1}" STREQUAL "STATIC")
        set(${_LIB_NAME}_LDFLAGS ${${_LIB_NAME}_MOD_STATIC_LDFLAGS})
      else("${ARGV1}" STREQUAL "STATIC")
        set(${_LIB_NAME}_LDFLAGS ${${_LIB_NAME}_MOD_LDFLAGS})
      endif("${ARGV1}" STREQUAL "STATIC")
      foreach(_FLAG ${${_LIB_NAME}_LDFLAGS})
        if(_FLAG MATCHES "^-l.*")
          string(REGEX REPLACE "^-l" "" _LIB ${_FLAG})
          list(APPEND ${_LIB_NAME}_LINK_LIBS ${_LIB})
        elseif(_FLAG MATCHES "^-L.*")
          string(REGEX REPLACE "^-L" "" _LIB_DIR ${_FLAG})
          list(APPEND ${_LIB_NAME}_LINK_DIRS ${_LIB_DIR})
        endif(_FLAG MATCHES "^-l.*")
      endforeach(_FLAG ${${_LIB_NAME}_LDFLAGS})

      add_definitions(${${_LIB_NAME}_DEFS})
      link_directories(${${_LIB_NAME}_LINK_DIRS})
      list(APPEND SMIL_EXT_DEPS ${${_LIB_NAME}_LINK_LIBS})

    endif(${_LIB_NAME}_MOD_FOUND)
  endif(PKG_CONFIG_FOUND)
endmacro(ADD_PKG_CONFIG_DEFS _MODULE)
