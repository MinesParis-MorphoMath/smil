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

SET(DEBUGON ON)
macro (pdebug msg)
  IF (DEBUGON)
    MESSAGE (${msg})
  ENDIF (DEBUGON)
endmacro (pdebug)

SET (CPACK_BUILD_BINARY_PACKAGES ON  CACHE BOOL "Build standard packages")
SET (CPACK_BUILD_DEVEL_PACKAGES  ON CACHE BOOL "Build devel packages")

FILE (GLOB NATIVE_LIB_DEPENDENCIES ${CMAKE_CURRENT_BINARY_DIR}/depends/*)

IF (NATIVE_LIB_DEPENDENCIES)
  INSTALL (FILES ${NATIVE_LIB_DEPENDENCIES} 
           DESTINATION ${SMIL_LIBRARIES_INSTALL_PATH} COMPONENT base)
  SET (SMIL_INSTALL_COMPONENT_LIST base ${SMIL_INSTALL_COMPONENT_LIST})
ENDIF (NATIVE_LIB_DEPENDENCIES)


INCLUDE(InstallRequiredSystemLibraries)

SET (CPACK_PACKAGE_NAME    "Smil")
SET (CPACK_PACKAGE_CONTACT "Jose-Marcio.Martins@mines-paristech.fr")

IF (WIN32)
  SET (DEFAULT_GENERATOR NSIS)
  IF (USE_64BIT_IDS)
    SET (CPACK_SYSTEM_NAME win64)
  ELSE (USE_64BIT_IDS)
    SET (CPACK_SYSTEM_NAME win32)
  ENDIF (USE_64BIT_IDS)
ELSE (WIN32)
  IF (CMAKE_CROSSCOMPILING)
    SET (DEFAULT_GENERATOR TGZ)
  ELSE (CMAKE_CROSSCOMPILING)
    SET (DEFAULT_GENERATOR DEB)
    IF (EXISTS "/etc/lsb-release")
      SET (DEFAULT_GENERATOR DEB)
    ELSEIF (EXISTS "/etc/redhat-release")
      SET (DEFAULT_GENERATOR RPM)
    ENDIF (EXISTS "/etc/lsb-release")

    IF (USE_64BIT_IDS)
      SET (CPACK_SYSTEM_NAME amd64)
    ELSE (USE_64BIT_IDS)
      SET (CPACK_SYSTEM_NAME i386)
    ENDIF (USE_64BIT_IDS)

    # If CPACK_DISTRIB_NAME is manually defined, add it to the system name
    # JOE - But how and where to manually define it ???
    IF (CPACK_DISTRIB_NAME)
      SET (CPACK_SYSTEM_NAME ${CPACK_DISTRIB_NAME}_${CPACK_SYSTEM_NAME})
    ENDIF (CPACK_DISTRIB_NAME)
  ENDIF (CMAKE_CROSSCOMPILING)
ENDIF (WIN32)


SET (CPACK_GENERATOR ${DEFAULT_GENERATOR} 
     CACHE STRING "CPack generator type (DEB, RPM, NSIS, TGZ, TBZ2, ZIP, ...)")


SET (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Simple Morphological Image Library")
SET (CPACK_PACKAGE_VENDOR 
    "Centre de Morphologie Mathematique - Mines-ParisTech")
# SET (CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
SET (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
SET (CPACK_PACKAGE_VERSION ${SMIL_VERSION})
SET (CPACK_PACKAGE_INSTALL_DIRECTORY "Smil")


SET (CPACK_SOURCE_IGNORE_FILES ".kdev*;.git;notes.txt;todo.txt")

#
# Select what to include in to the packages
SET (CPACK_COMPONENTS_ALL)
#
FOREACH (COMP ${SMIL_INSTALL_COMPONENT_LIST})
  LIST_CONTAINS(IS_IN_LIST ${COMP} ${CPACK_COMPONENTS_ALL})
  IF (NOT IS_IN_LIST)
    STRING(TOUPPER ${COMP} COMP_UP)
    STRING(REGEX REPLACE "-.*$" "" COMP_PREFIX ${COMP})
    STRING(REGEX REPLACE ".*-" ""  COMP_SUFFIX ${COMP})

    IF (COMP_SUFFIX STREQUAL "dev" AND CPACK_BUILD_DEVEL_PACKAGES)
      LIST(APPEND CPACK_COMPONENTS_ALL ${COMP})
    ELSEIF (NOT COMP_SUFFIX STREQUAL "dev" AND CPACK_BUILD_BINARY_PACKAGES)
      LIST(APPEND CPACK_COMPONENTS_ALL ${COMP})
    ENDIF ()

    LIST_CONTAINS (IS_ADDON ${COMP_PREFIX} ${AVAILABLE_ADDONS})
    IF (IS_ADDON)
      SET (CPACK_COMPONENT_${COMP_UP}_DEPENDS ${COMP_SUFFIX})
    ENDIF (IS_ADDON)
  ENDIF (NOT IS_IN_LIST)
ENDFOREACH (COMP ${SMIL_COMPONENT_LIST})

SET (CPACK_COMPONENT_BASE_DISPLAY_NAME   "Smil base libraries")
SET (CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Smil python libraries")
SET (CPACK_COMPONENT_JAVA_DISPLAY_NAME   "Smil java libraries")

# SET (CPACK_COMPONENTS_IGNORE_GROUPS TRUE)

IF (BUILD_SHARED_LIBS OR NATIVE_LIB_DEPENDENCIES)
  SET (CPACK_COMPONENT_BASE_REQUIRED TRUE)
  SET (CPACK_COMPONENT_PYTHON_DEPENDS base)
  SET (CPACK_COMPONENT_JAVA_DEPENDS base)
ENDIF (BUILD_SHARED_LIBS OR NATIVE_LIB_DEPENDENCIES)


#### NSIS ####
IF (CPACK_GENERATOR MATCHES "NSIS")
  #  SET (CPACK_GENERATOR NSIS)
  SET (CPACK_NSIS_COMPONENT_INSTALL ON)
  SET (CPACK_NSIS_PACKAGE_COMPONENT TRUE)
  SET (CPACK_NSIS_DISPLAY_NAME      "Smil")
  #  SET (CPACK_TOPLEVEL_DIRECTORY "${CMAKE_BINARY_DIR}/_CPack_Packages/win32/NSIS")
  SET (CPACK_NSIS_HELP_LINK      "https://smil.cmm.minesparis.psl.eu/doc/")
  SET (CPACK_NSIS_URL_INFO_ABOUT "https://smil.cmm.minesparis.psl.eu/doc/")

  IF (WRAP_PYTHON)
    IF (USE_64BIT_IDS)
      # allow the installer to access keys in the x64 view of the registry
      SET (CPACK_NSIS_EXTRA_INSTALL_COMMANDS "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS} 
			      SetRegView 64")
    ENDIF (USE_64BIT_IDS)
    SET (CPACK_NSIS_EXTRA_INSTALL_COMMANDS "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS}
			WriteRegStr HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${PYTHON_VERSION}\\\\PythonPath\\\\Smil\\\" \\\"\\\" \\\"$INSTDIR\\\\${SMIL_LIBRARIES_INSTALL_PATH}\\\"
	      ")

    SET (CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "${CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS}
			DeleteRegValue HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${PYTHON_VERSION}\\\\PythonPath\\\" \\\"Smil\\\"
	      ")
  ENDIF (WRAP_PYTHON)
ENDIF (CPACK_GENERATOR MATCHES "NSIS")

### CHECK PYTHON VERSION ###
IF (WRAP_PYTHON)
  SET (PYTHON_MAJOR   2)
  IF (PYTHON_VERSION)
    STRING (REGEX MATCH "[0-9]" PYTHON_MAJOR "${PYTHON_VERSION}")
  ENDIF ()
ENDIF (WRAP_PYTHON)

#### DEB ####
IF (CPACK_GENERATOR MATCHES "DEB")
  OPTION(CPACK_USE_SHLIBDEPS "Use shlibdeps to determine package dependencies. 
	  This requires to add the libraries path to LD_LIBRARY_PATH" TRUE)
	  
  SET (CPACK_DEB_PACKAGE_COMPONENT TRUE)
  SET (CPACK_DEB_COMPONENT_INSTALL TRUE)

  SET (CPACK_DEBIAN_PACKAGE_MAINTAINER 
      "Jose-Marcio Martins da Cruz <Jose-Marcio.Martins@mines-paristech.fr>")
  IF (USE_64BIT_IDS)
    SET (CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
  ELSE (USE_64BIT_IDS)
    SET (CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
  ENDIF (USE_64BIT_IDS)

  SET (CPACK_DEB_COMPONENT_INSTALL ON)

  IF (CPACK_USE_SHLIBDEPS)
    SET (CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  ELSE (CPACK_USE_SHLIBDEPS)
    # JOE - Not sure this works !!!
    SET (CPACK_DEBIAN_COMPONENT_base_DEPENDS "${CPACK_DEB_COMMON_DEPENDS}")
	  SET (CPACK_DEBIAN_COMPONENT_python_DEPENDS 
         "${CPACK_DEB_COMMON_DEPENDS}, python${PYTHON_VERSION}")
  ENDIF (CPACK_USE_SHLIBDEPS)

  SET (CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

  IF (EXISTS "/usr/bin/lsb_release")
    SET (DEFAULT_GENERATOR DEB)
    EXECUTE_PROCESS(COMMAND "/usr/bin/lsb_release -s -c" OUTPUT_VARIABLE DISTRIBUTION)
    SET (CPACK_DEBIAN_PACKAGE_RELEASE "${DISTRIBUTION}") 
  ENDIF (EXISTS "/usr/bin/lsb_release")
ENDIF (CPACK_GENERATOR MATCHES "DEB")      

#### RPM ####
IF (CPACK_GENERATOR MATCHES "RPM")
  SET (CPACK_RPM_COMPONENT_INSTALL ON)

  IF (USE_64BIT_IDS)
    SET (CPACK_RPM_PACKAGE_ARCHITECTURE x86_64)
    SET (CPACK_SYSTEM_NAME              x86_64)
  ELSE (USE_64BIT_IDS)
    SET (CPACK_RPM_PACKAGE_ARCHITECTURE i386)
    SET (CPACK_SYSTEM_NAME              i386)
  ENDIF (USE_64BIT_IDS)

  SET (LIB_DEPS)
  SET (LIB_DEPS, "libtiff, libjpeg-turbo")
  IF (USE_NUMPY)
    #SET (LIB_DEPS "${LIB_DEPS}, python${PYTHON_MAJOR}-numpy")
  ENDIF(USE_NUMPY)
  IF (USE_QT)
    IF (USE_QT_VERSION STREQUAL "5")
      SET (LIB_DEPS "${LIB_DEPS}, qt5-qtbase")
    ENDIF ()

    # SET (LIB_DEPS "${LIB_DEPS}, qt >= ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}")
    IF (USE_QWT)
    #	SET (LIB_DEPS "${LIB_DEPS}, qwt")
    ENDIF (USE_QWT)
  ENDIF (USE_QT)

  # set(CPACK_RPM_PACKAGE_REQUIRES "python >= 2.5.0, cmake >= 2.8")
  SET (CPACK_RPM_PACKAGE_REQUIRES ${LIB_DEPS})    

  LIST(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local")
  LIST(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib64")
  LIST(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/include")
  LIST(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/lib64/pkgconfig")
ENDIF (CPACK_GENERATOR MATCHES "RPM")      



#### TGZ ####
IF (CPACK_GENERATOR MATCHES "TGZ")
ENDIF (CPACK_GENERATOR MATCHES "TGZ")

INCLUDE(CPack)

