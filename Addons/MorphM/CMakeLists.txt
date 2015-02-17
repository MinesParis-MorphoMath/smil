# Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


SET(MODULE_NAME MorphM)

FIND_PATH(MORPHEE_INCLUDE_DIR morphee/image/include/private/image_T.hpp
	  DOC "Morphee sources directory"
)
FIND_PATH(MORPHEE_LIBRARY_DIR NAMES libMorpheecommon.so Morpheecommon.dll
	  DOC "Morphee libraries directory"
)

INCLUDE_DIRECTORIES(${MORPHEE_INCLUDE_DIR})

ADD_DEFINITIONS(-DMORPHEE_LIBRARY_DIR="${MORPHEE_LIBRARY_DIR}")
SET(MODULE_DEPS ${SMIL_LIB_PREFIX}Core -L"${MORPHEE_LIBRARY_DIR}" Morpheecommon Morpheeimage boost_python)

IF(WRAP_PYTHON)
    LIST(APPEND MODULE_DEPS ${PYTHON_LIBRARIES})
    ADD_DEFINITIONS(-DWRAP_PYTHON)
ENDIF(WRAP_PYTHON)

IF(USE_64BIT_IDS)
    ADD_DEFINITIONS(-DHAVE_64BITS)
ENDIF(USE_64BIT_IDS)


ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS})
ADD_SMIL_TESTS(${MODULE_NAME} ${MODULE_DEPS})


