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
# Toolchain configuration for clang/llvm

# SET(CMAKE_C_COMPILER clang)
# SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_COMPILER llvm-gcc)
SET(CMAKE_CXX_COMPILER llvm-g++)


# SET(CMAKE_TOOLCHAIN_PREFIX llvm-)

# SET(CMAKE_AR "/usr/bin/llvm-ar --plugin libLLVMgold.so" CACHE FILEPATH "llvm ar")
SET(CMAKE_AR "/usr/bin/llvm-ar" CACHE FILEPATH "llvm ar")
SET(CMAKE_LINKER "/usr/bin/llvm-ld" CACHE FILEPATH "llvm ld")
SET(CMAKE_NM "/usr/bin/llvm-nm" CACHE FILEPATH "llvm nm")
SET(CMAKE_OBJDUMP "/usr/bin/llvm-objdump" CACHE FILEPATH "llvm objdump")
# SET(CMAKE_RANLIB "/usr/bin/llvm-ranlib" CACHE FILEPATH "llvm ranlib")
SET(CMAKE_RANLIB "/bin/true" CACHE FILEPATH "llvm ranlib")

# SET (CMAKE_C_FLAGS "-std=c99" CACHE STRING "")
SET (CMAKE_C_FLAGS "" CACHE STRING "")
SET (CMAKE_C_FLAGS_DEBUG "-g" CACHE STRING "")
SET (CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "")
SET (CMAKE_C_FLAGS_RELEASE "-O4 -DNDEBUG" CACHE STRING "")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")

# SET (CMAKE_CXX_FLAGS "-Wall" CACHE STRING "")
SET (CMAKE_CXX_FLAGS "" CACHE STRING "")
SET (CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "")
SET (CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "")
# Requires llvm-dev and binutils-gold
# SET (CMAKE_CXX_FLAGS_RELEASE "-use-gold-plugin -O4 -DNDEBUG" CACHE STRING "")
SET (CMAKE_CXX_FLAGS_RELEASE "-O4 -DNDEBUG" CACHE STRING "")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")


