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

set(DEBUGON ON)
macro(pdebug msg)
  if(DEBUGON)
    message(${msg})
  endif(DEBUGON)
endmacro(pdebug)

set(CPACK_BUILD_BINARY_PACKAGES
    ON
    CACHE BOOL "Build standard packages")
set(CPACK_BUILD_DEVEL_PACKAGES
    ON
    CACHE BOOL "Build devel packages")

file(GLOB NATIVE_LIB_DEPENDENCIES ${CMAKE_CURRENT_BINARY_DIR}/depends/*)

if(NATIVE_LIB_DEPENDENCIES)
  install(
    FILES ${NATIVE_LIB_DEPENDENCIES}
    DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH}
    COMPONENT base)
  set(SMIL_INSTALL_COMPONENT_LIST base ${SMIL_INSTALL_COMPONENT_LIST})
endif(NATIVE_LIB_DEPENDENCIES)

include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_NAME "Smil")
set(CPACK_PACKAGE_CONTACT "Jose-Marcio.Martins@mines-paristech.fr")

if(WIN32)
  set(DEFAULT_GENERATOR NSIS)
  if(USE_64BIT_IDS)
    set(CPACK_SYSTEM_NAME win64)
  else(USE_64BIT_IDS)
    set(CPACK_SYSTEM_NAME win32)
  endif(USE_64BIT_IDS)
else(WIN32)
  if(CMAKE_CROSSCOMPILING)
    set(DEFAULT_GENERATOR TGZ)
  else(CMAKE_CROSSCOMPILING)
    set(DEFAULT_GENERATOR DEB)
    if(EXISTS "/etc/lsb-release")
      set(DEFAULT_GENERATOR DEB)
    elseif(EXISTS "/etc/redhat-release")
      set(DEFAULT_GENERATOR RPM)
    endif(EXISTS "/etc/lsb-release")

    if(USE_64BIT_IDS)
      set(CPACK_SYSTEM_NAME amd64)
    else(USE_64BIT_IDS)
      set(CPACK_SYSTEM_NAME i386)
    endif(USE_64BIT_IDS)

    # If CPACK_DISTRIB_NAME is manually defined, add it to the system name JOE -
    # But how and where to manually define it ???
    if(CPACK_DISTRIB_NAME)
      set(CPACK_SYSTEM_NAME ${CPACK_DISTRIB_NAME}_${CPACK_SYSTEM_NAME})
    endif(CPACK_DISTRIB_NAME)
  endif(CMAKE_CROSSCOMPILING)
endif(WIN32)

set(CPACK_GENERATOR
    ${DEFAULT_GENERATOR}
    CACHE STRING "CPack generator type (DEB, RPM, NSIS, TGZ, TBZ2, ZIP, ...)")

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Simple Morphological Image Library")
set(CPACK_PACKAGE_VENDOR "Centre de Morphologie Mathematique - Mines-ParisTech")
# SET (CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_PACKAGE_VERSION ${SMIL_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Smil")

set(CPACK_SOURCE_IGNORE_FILES ".kdev*;.git;notes.txt;todo.txt")

#
# Select what to include in to the packages
set(CPACK_COMPONENTS_ALL)
#
foreach(COMP ${SMIL_INSTALL_COMPONENT_LIST})
  list_contains(IS_IN_LIST ${COMP} ${CPACK_COMPONENTS_ALL})
  if(NOT IS_IN_LIST)
    string(TOUPPER ${COMP} COMP_UP)
    string(REGEX REPLACE "-.*$" "" COMP_PREFIX ${COMP})
    string(REGEX REPLACE ".*-" "" COMP_SUFFIX ${COMP})

    if(COMP_SUFFIX STREQUAL "dev" AND CPACK_BUILD_DEVEL_PACKAGES)
      list(APPEND CPACK_COMPONENTS_ALL ${COMP})
    elseif(NOT COMP_SUFFIX STREQUAL "dev" AND CPACK_BUILD_BINARY_PACKAGES)
      list(APPEND CPACK_COMPONENTS_ALL ${COMP})
    endif()

    list_contains(IS_ADDON ${COMP_PREFIX} ${AVAILABLE_ADDONS})
    if(IS_ADDON)
      set(CPACK_COMPONENT_${COMP_UP}_DEPENDS ${COMP_SUFFIX})
    endif(IS_ADDON)
  endif(NOT IS_IN_LIST)
endforeach(COMP ${SMIL_COMPONENT_LIST})

set(CPACK_COMPONENT_BASE_DISPLAY_NAME "Smil base libraries")
set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Smil python libraries")
set(CPACK_COMPONENT_JAVA_DISPLAY_NAME "Smil java libraries")

# SET (CPACK_COMPONENTS_IGNORE_GROUPS TRUE)

if(BUILD_SHARED_LIBS OR NATIVE_LIB_DEPENDENCIES)
  set(CPACK_COMPONENT_BASE_REQUIRED TRUE)
  set(CPACK_COMPONENT_PYTHON_DEPENDS base)
  set(CPACK_COMPONENT_JAVA_DEPENDS base)
endif(BUILD_SHARED_LIBS OR NATIVE_LIB_DEPENDENCIES)

# NSIS ####
if(CPACK_GENERATOR MATCHES "NSIS")
  # SET (CPACK_GENERATOR NSIS)
  set(CPACK_NSIS_COMPONENT_INSTALL ON)
  set(CPACK_NSIS_PACKAGE_COMPONENT TRUE)
  set(CPACK_NSIS_DISPLAY_NAME "Smil")
  # SET (CPACK_TOPLEVEL_DIRECTORY
  # "${CMAKE_BINARY_DIR}/_CPack_Packages/win32/NSIS")
  set(CPACK_NSIS_HELP_LINK "https://smil.cmm.minesparis.psl.eu/doc/")
  set(CPACK_NSIS_URL_INFO_ABOUT "https://smil.cmm.minesparis.psl.eu/doc/")

  if(WRAP_PYTHON)
    if(USE_64BIT_IDS)
      # allow the installer to access keys in the x64 view of the registry
      set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
          "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS}
			      SetRegView 64")
    endif(USE_64BIT_IDS)
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
        "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS}
			WriteRegStr HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${PYTHON_VERSION}\\\\PythonPath\\\\Smil\\\" \\\"\\\" \\\"$INSTDIR\\\\${SMIL_LIBRARIES_INSTALL_PATH}\\\"
	      ")

    set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS
        "${CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS}
			DeleteRegValue HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${PYTHON_VERSION}\\\\PythonPath\\\" \\\"Smil\\\"
	      ")
  endif(WRAP_PYTHON)
endif(CPACK_GENERATOR MATCHES "NSIS")

# CHECK PYTHON VERSION ###
if(WRAP_PYTHON)
  set(PYTHON_MAJOR 2)
  if(PYTHON_VERSION)
    string(REGEX MATCH "[0-9]" PYTHON_MAJOR "${PYTHON_VERSION}")
  endif()
endif(WRAP_PYTHON)

# DEB ####
if(CPACK_GENERATOR MATCHES "DEB")
  option(CPACK_USE_SHLIBDEPS "Use shlibdeps to determine package dependencies.
	  This requires to add the libraries path to LD_LIBRARY_PATH" TRUE)

  set(CPACK_DEB_PACKAGE_COMPONENT TRUE)
  set(CPACK_DEB_COMPONENT_INSTALL TRUE)

  set(CPACK_DEBIAN_PACKAGE_MAINTAINER
      "Jose-Marcio Martins da Cruz <Jose-Marcio.Martins@mines-paristech.fr>")
  if(USE_64BIT_IDS)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
  else(USE_64BIT_IDS)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
  endif(USE_64BIT_IDS)

  set(CPACK_DEB_COMPONENT_INSTALL ON)

  if(CPACK_USE_SHLIBDEPS)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  else(CPACK_USE_SHLIBDEPS)
    # JOE - Not sure this works !!!
    set(CPACK_DEBIAN_COMPONENT_base_DEPENDS "${CPACK_DEB_COMMON_DEPENDS}")
    set(CPACK_DEBIAN_COMPONENT_python_DEPENDS
        "${CPACK_DEB_COMMON_DEPENDS}, python${PYTHON_VERSION}")
  endif(CPACK_USE_SHLIBDEPS)

  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

  if(EXISTS "/usr/bin/lsb_release")
    set(DEFAULT_GENERATOR DEB)
    execute_process(COMMAND "/usr/bin/lsb_release -s -c"
                    OUTPUT_VARIABLE DISTRIBUTION)
    set(CPACK_DEBIAN_PACKAGE_RELEASE "${DISTRIBUTION}")
  endif(EXISTS "/usr/bin/lsb_release")
endif(CPACK_GENERATOR MATCHES "DEB")

# RPM ####
if(CPACK_GENERATOR MATCHES "RPM")
  set(CPACK_RPM_COMPONENT_INSTALL ON)

  if(USE_64BIT_IDS)
    set(CPACK_RPM_PACKAGE_ARCHITECTURE x86_64)
    set(CPACK_SYSTEM_NAME x86_64)
  else(USE_64BIT_IDS)
    set(CPACK_RPM_PACKAGE_ARCHITECTURE i386)
    set(CPACK_SYSTEM_NAME i386)
  endif(USE_64BIT_IDS)

  set(LIB_DEPS)
  set(LIB_DEPS, "libtiff, libjpeg-turbo")
  if(USE_NUMPY)
    # SET (LIB_DEPS "${LIB_DEPS}, python${PYTHON_MAJOR}-numpy")
  endif(USE_NUMPY)
  if(USE_QT)
    if(USE_QT_VERSION STREQUAL "5")
      set(LIB_DEPS "${LIB_DEPS}, qt5-qtbase")
    endif()

    # SET (LIB_DEPS "${LIB_DEPS}, qt >=
    # ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}")
    if(USE_QWT)
      # SET (LIB_DEPS "${LIB_DEPS}, qwt")
    endif(USE_QWT)
  endif(USE_QT)

  # set(CPACK_RPM_PACKAGE_REQUIRES "python >= 2.5.0, cmake >= 2.8")
  set(CPACK_RPM_PACKAGE_REQUIRES ${LIB_DEPS})

  list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local")
  list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib64")
  list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
       "/usr/local/include")
  list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
       "/usr/lib64/pkgconfig")
endif(CPACK_GENERATOR MATCHES "RPM")

# TGZ ####
if(CPACK_GENERATOR MATCHES "TGZ")

endif(CPACK_GENERATOR MATCHES "TGZ")

include(CPack)
