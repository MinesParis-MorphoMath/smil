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

SET(MODULE_NAME Vtk)

FIND_PACKAGE(VTK REQUIRED)

IF(VTK_FOUND)
        INCLUDE(${VTK_USE_FILE})
        LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})
        LIST(APPEND CMAKE_MODULE_PATH ${VTK_CMAKE_DIR})
ENDIF(VTK_FOUND)

SET(MODULE_DEPS ${SMIL_LIB_PREFIX}Core vtkImaging vtkIO)

IF(WRAP_PYTHON)
      FIND_LIBRARY(VTK_PYTHON_CORE_LIB NAMES vtkPythonCore)
      LIST(APPEND MODULE_DEPS ${VTK_PYTHON_CORE_LIB})
ENDIF(WRAP_PYTHON)


ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS})
ADD_SMIL_TESTS(${MODULE_NAME} ${MODULE_DEPS})


