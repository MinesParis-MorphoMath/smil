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

INCLUDE(InstallRequiredSystemLibraries)

SET(CPACK_PACKAGE_NAME "Smil")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Simple Morphological Image Library")
SET(CPACK_PACKAGE_VENDOR "Matthieu Faessel and ARMINES")
# SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/license.txt")
SET(CPACK_PACKAGE_VERSION_MAJOR ${SMIL_MAJOR_VERSION})
SET(CPACK_PACKAGE_VERSION_MINOR ${SMIL_MINOR_VERSION})
SET(CPACK_PACKAGE_VERSION_PATCH ${SMIL_PATCH_VERSION})
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "Smil")
IF(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
#   SET(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
#   SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
#   SET(CPACK_NSIS_DISPLAY_NAME "Smil")
      SET(CPACK_NSIS_HELP_LINK "http://cmm.ensmp.fr/~faessel/smil/doc/")
      SET(CPACK_NSIS_URL_INFO_ABOUT "http://cmm.ensmp.fr/~faessel/smil/doc/")
    #   SET(CPACK_NSIS_CONTACT "me@my-personal-home-page.com")
      SET(CPACK_NSIS_MODIFY_PATH ON)
ELSE(WIN32 AND NOT UNIX)
    #   SET(CPACK_STRIP_FILES "bin/MyExecutable")
#       SET(CPACK_SOURCE_STRIP_FILES "")
      SET(CPACK_GENERATOR "DEB")
      SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Matthieu Faessel") #required
      IF(USE_64BIT_IDS)
	      SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
      ELSE(USE_64BIT_IDS)
	      SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
      ENDIF(USE_64BIT_IDS)
      SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.4), libgcc1 (>= 1:3.0)")
ENDIF(WIN32 AND NOT UNIX)
# SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE(CPack)
