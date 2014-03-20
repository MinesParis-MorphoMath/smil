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


SET(MODULE_NAME IO)
SET(MODULE_DEPS smilCore smilGui)

SET(EXCL_SRCS)
IF(NOT USE_PNG)
	LIST(APPEND EXCL_SRCS DImageIO_PNG.cpp)
ENDIF(NOT USE_PNG)
IF(NOT USE_JPEG)
	LIST(APPEND EXCL_SRCS DImageIO_JPG.cpp)
ENDIF(NOT USE_JPEG)
IF(NOT USE_TIFF)
	LIST(APPEND EXCL_SRCS DImageIO_TIFF.cpp)
ENDIF(NOT USE_TIFF)

ADD_SMIL_LIBRARY(${MODULE_NAME} ${MODULE_DEPS} EXCLUDED_SOURCES ${EXCL_SRCS})
ADD_SMIL_TESTS(${MODULE_NAME} ${MODULE_DEPS} smilCore)
