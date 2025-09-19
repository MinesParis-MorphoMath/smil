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

set(AVAILABLE_ADDONS)
file(
  GLOB _ADD_CMLISTS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/"
  "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
foreach(_FCH ${_ADD_CMLISTS})
  get_filename_component(_DIR ${_FCH} PATH)
  string(TOUPPER ${_DIR} _ADDON)
  list(APPEND AVAILABLE_ADDONS ${_DIR})
  option(ADDON_${_ADDON} "Build addon ${_DIR}" OFF)
  mark_as_advanced(ADDON_${_ADDON})
  if(ADDON_${_ADDON})
    include_directories("${_DIR}" "${_DIR}/include" "${_DIR}/include/private")
    add_subdirectory("${_DIR}")
  endif(ADDON_${_ADDON})
endforeach(_FCH ${_ADD_CMLISTS})

set(SMIL_INSTALL_COMPONENT_LIST
    ${SMIL_INSTALL_COMPONENT_LIST}
    PARENT_SCOPE)
set(AVAILABLE_ADDONS
    ${AVAILABLE_ADDONS}
    PARENT_SCOPE)
