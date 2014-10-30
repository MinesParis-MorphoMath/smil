# Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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

SET(AVAILABLE_USER_MODULES)

FILE(GLOB _MOD_CMLISTS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
FOREACH(_FCH ${_MOD_CMLISTS})
	GET_FILENAME_COMPONENT(_DIR ${_FCH} PATH)
	STRING(TOUPPER ${_DIR} _USER_MOD)
	LIST(APPEND AVAILABLE_USER_MODS ${_DIR})
	OPTION(ADD_USER_MOD_${_USER_MOD} "Build additional user module ${_DIR}" OFF)
	MARK_AS_ADVANCED(ADDON_${_USER_MOD})
	IF(ADD_USER_MOD_${_USER_MOD})
		INCLUDE_DIRECTORIES("${_DIR}" "${_DIR}/include" "${_DIR}/include/private")
		ADD_SUBDIRECTORY("${_DIR}")
	ENDIF(ADD_USER_MOD_${_USER_MOD})
ENDFOREACH(_FCH ${_MOD_CMLISTS})
