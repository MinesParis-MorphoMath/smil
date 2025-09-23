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

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
macro(AddCompilerFlag _flag)
  string(REGEX REPLACE "[+/:= ]" "_" _flag_esc "${_flag}")
  check_c_compiler_flag("${_flag}" check_c_compiler_flag_${_flag_esc})
  check_cxx_compiler_flag("${_flag}" check_cxx_compiler_flag_${_flag_esc})
  if(check_c_compiler_flag_${_flag_esc})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_flag}")
  endif(check_c_compiler_flag_${_flag_esc})
  if(check_cxx_compiler_flag_${_flag_esc})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_flag}")
  endif(check_cxx_compiler_flag_${_flag_esc})
  if(${ARGC} EQUAL 2)
    set(${ARGV1} "${check_cxx_compiler_flag_${_flag_esc}}")
  endif(${ARGC} EQUAL 2)
endmacro(AddCompilerFlag)

set(CPU_VENDOR_ID 0)

macro(GET_CPU_INFOS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(READ /proc/cpuinfo cpuinfo)
    string(REGEX REPLACE ".*\nprocessor[ \t]*:[ \t]+([^\n\r]+).*" "\\1" CPU_NBR
                         "${cpuinfo}")
    math(EXPR PROC_NBR "${CPU_NBR}+1")
    string(REGEX REPLACE ".*\nvendor_id[ \t]*:[ \t]+([^\n]+).*" "\\1"
                         CPU_VENDOR_ID "${cpuinfo}")
    string(REGEX REPLACE ".*\ncpu family[ \t]*:[ \t]+([^\n]+).*" "\\1"
                         CPU_FAMILY "${cpuinfo}")
    string(REGEX REPLACE ".*\nmodel[ \t]*:[ \t]+([^\n]+).*" "\\1" CPU_MODEL
                         "${cpuinfo}")
    string(REGEX REPLACE ".*\nmodel name[ \t]*:[ \t]+([^\n]+).*" "\\1"
                         CPU_MODEL_NAME "${cpuinfo}")
    string(REGEX REPLACE ".*[@ \t]+([^ \t]+).*" "\\1" CPU_SPEED
                         "${CPU_MODEL_NAME}")
    string(REGEX REPLACE ".*\ncache size[ \t]*:[ \t]+([^\n ]+).*" "\\1"
                         CPU_CACHE_SIZE "${cpuinfo}")
    string(REGEX REPLACE ".*\nflags[ \t]*:[ \t]+([^\n]+).*" "\\1" CPU_FLAGS
                         "${cpuinfo}")
    string(REGEX REPLACE " " ";" CPU_FLAGS "${CPU_FLAGS}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    exec_program("/usr/sbin/sysctl -n machdep.cpu.vendor" OUTPUT_VARIABLE
                 CPU_VENDOR_ID)
    exec_program("/usr/sbin/sysctl -n machdep.cpu.model" OUTPUT_VARIABLE
                 CPU_MODEL)
    exec_program("/usr/sbin/sysctl -n machdep.cpu.family" OUTPUT_VARIABLE
                 CPU_FAMILY)
    exec_program("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE
                 CPU_FLAGS)
    string(TOLOWER "${CPU_FLAGS}" CPU_FLAGS)
    string(REPLACE "." "_" CPU_FLAGS "${CPU_FLAGS}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    get_filename_component(
      CPU_VENDOR_ID
      "[HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0;VendorIdentifier]"
      NAME)
    get_filename_component(
      cpuinfo
      "[HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0;Identifier]"
      NAME)
    string(REGEX REPLACE ".* Family ([0-9]+) .*" "\\1" CPU_FAMILY "${cpuinfo}")
    string(REGEX REPLACE ".* Model ([0-9]+) .*" "\\1" CPU_MODEL "${cpuinfo}")
  endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
endmacro(GET_CPU_INFOS)

# MESSAGE("CPU NBR: ${CPU_NBR}") MESSAGE("CPU_VENDOR_ID: ${CPU_VENDOR_ID}")
# MESSAGE("CPU_FAMILY: ${CPU_FAMILY}") MESSAGE("CPU_MODEL: ${CPU_MODEL}")
# MESSAGE("CPU_MODEL_NAME: ${CPU_MODEL_NAME}") MESSAGE("CPU_SPEED:
# ${CPU_SPEED}") MESSAGE("CPU_CACHE_SIZE: ${CPU_CACHE_SIZE}")
# MESSAGE("CPU_FLAGS: ${CPU_FLAGS}")

set(CPU_ARCHITECTURES
    # VENDOR_ID               FAMILY  MODEL   ARCH
    "GenuineIntel	6	14	core"
    "GenuineIntel	6	15	merom"
    "GenuineIntel	6	23	penryn"
    "GenuineIntel	6	26	nehalem"
    "GenuineIntel	6	28	atom"
    "GenuineIntel	6	29	penryn"
    "GenuineIntel	6	30	westmere"
    "GenuineIntel	6	31	westmere"
    "GenuineIntel	6	37	westmere"
    "GenuineIntel	6	42	sandy-bridge"
    "GenuineIntel	6	44	westmere"
    "GenuineIntel	6	45	sandy-bridge"
    "GenuineIntel	6	46	westmere"
    "AuthenticAMD	15	*	k8"
    "AuthenticAMD	16	*	barcelona"
    "AuthenticAMD	21	*	bulldozer")

set(ARCH_FLAGS "core")
# list(APPEND _march_flag_list "core2") list(APPEND _available_vector_units_list
# "sse" "sse2" "sse3") elseif(TARGET_ARCHITECTURE STREQUAL "merom") list(APPEND
# _march_flag_list "merom") list(APPEND _march_flag_list "core2") list(APPEND
# _available_vector_units_list "sse" "sse2" "sse3" "ssse3")
# elseif(TARGET_ARCHITECTURE STREQUAL "penryn") list(APPEND _march_flag_list
# "penryn") list(APPEND _march_flag_list "core2") list(APPEND
# _available_vector_units_list "sse" "sse2" "sse3" "ssse3") message(STATUS
# "Sadly the Penryn architecture exists in variants with SSE4.1 and without
# SSE4.1.") if(_cpu_flags MATCHES "sse4_1") message(STATUS "SSE4.1: enabled
# (auto-detected from this computer's CPU flags)") list(APPEND
# _available_vector_units_list "sse4.1") else() message(STATUS "SSE4.1: disabled
# (auto-detected from this computer's CPU flags)") endif()
# elseif(TARGET_ARCHITECTURE STREQUAL "nehalem") list(APPEND _march_flag_list
# "nehalem") list(APPEND _march_flag_list "corei7") list(APPEND _march_flag_list
# "core2") list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3"
# "sse4.1" "sse4.2") elseif(TARGET_ARCHITECTURE STREQUAL "westmere") list(APPEND
# _march_flag_list "westmere") list(APPEND _march_flag_list "corei7")
# list(APPEND _march_flag_list "core2") list(APPEND _available_vector_units_list
# "sse" "sse2" "sse3" "ssse3" "sse4.1" "sse4.2") elseif(TARGET_ARCHITECTURE
# STREQUAL "sandy-bridge") list(APPEND _march_flag_list "sandybridge")
# list(APPEND _march_flag_list "corei7-avx") list(APPEND _march_flag_list
# "core2") list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3"
# "sse4.1" "sse4.2" "avx") elseif(TARGET_ARCHITECTURE STREQUAL "atom")
# list(APPEND _march_flag_list "atom") list(APPEND _march_flag_list "core2")
# list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3")
# elseif(TARGET_ARCHITECTURE STREQUAL "k8") list(APPEND _march_flag_list "k8")
# list(APPEND _available_vector_units_list "sse" "sse2")
# elseif(TARGET_ARCHITECTURE STREQUAL "k8-sse3") list(APPEND _march_flag_list
# "k8-sse3") list(APPEND _march_flag_list "k8") list(APPEND
# _available_vector_units_list "sse" "sse2" "sse3") elseif(TARGET_ARCHITECTURE
# STREQUAL "interlagos") list(APPEND _march_flag_list "bulldozer") list(APPEND
# _march_flag_list "barcelona") list(APPEND _march_flag_list "core2")
# list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a"
# "sse4.1" "sse4.2" "avx" "xop" "fma4") elseif(TARGET_ARCHITECTURE STREQUAL
# "bulldozer") list(APPEND _march_flag_list "bulldozer") list(APPEND
# _march_flag_list "barcelona") list(APPEND _march_flag_list "core2")
# list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "ssse3" "sse4a"
# "sse4.1" "sse4.2" "avx" "xop" "fma4") elseif(TARGET_ARCHITECTURE STREQUAL
# "barcelona") list(APPEND _march_flag_list "barcelona") list(APPEND
# _march_flag_list "core2") list(APPEND _available_vector_units_list "sse"
# "sse2" "sse3" "sse4a") elseif(TARGET_ARCHITECTURE STREQUAL "istanbul")
# list(APPEND _march_flag_list "barcelona") list(APPEND _march_flag_list
# "core2") list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "sse4a")
# elseif(TARGET_ARCHITECTURE STREQUAL "magny-cours") list(APPEND
# _march_flag_list "barcelona") list(APPEND _march_flag_list "core2")
# list(APPEND _available_vector_units_list "sse" "sse2" "sse3" "sse4a")
# elseif(TARGET_ARCHITECTURE STREQUAL "generic") list(APPEND _march_flag_list
# "generic") elseif(TARGET_ARCHITECTURE STREQUAL "none") # add this clause to
# remove it from the else clause else(TARGET_ARCHITECTURE STREQUAL "core")
# message(FATAL_ERROR "Unknown target architecture: \"${TARGET_ARCHITECTURE}\".
# Please set TARGET_ARCHITECTURE to a supported value.")
# endif(TARGET_ARCHITECTURE STREQUAL "core")

macro(GET_CPU_ARCH)
  set(CPU_ARCH 0)
  if(NOT CPU_VENDOR_ID)
    get_cpu_infos()
  endif(NOT CPU_VENDOR_ID)
  foreach(_ARCH ${CPU_ARCHITECTURES})
    string(REGEX REPLACE "([^ \t]+).*" "\\1" _VENDOR_ID ${_ARCH})
    string(REGEX REPLACE "[^ \t]+\t([^ \t]+).*" "\\1" _FAMILY ${_ARCH})
    string(REGEX REPLACE "[^ \t]+\t[^ \t]+\t([^ \t]+).*" "\\1" _MODEL ${_ARCH})
    string(REGEX REPLACE "[^ \t]+\t[^ \t]+\t[^ \t]+\t([^ \t]+).*" "\\1"
                         _CPU_ARCH ${_ARCH})
    if(_VENDOR_ID STREQUAL ${CPU_VENDOR_ID})
      if(_FAMILY EQUAL ${CPU_FAMILY})
        if(_MODEL EQUAL ${CPU_MODEL} OR _MODEL STREQUAL "*")
          set(CPU_ARCH ${_CPU_ARCH})
        endif(_MODEL EQUAL ${CPU_MODEL} OR _MODEL STREQUAL "*")
      endif(_FAMILY EQUAL ${CPU_FAMILY})
    endif(_VENDOR_ID STREQUAL ${CPU_VENDOR_ID})
  endforeach(_ARCH ${CPU_ARCHITECTURES})
endmacro(GET_CPU_ARCH)

get_cpu_arch()

message("CPU_ARCH: ${CPU_ARCH}")
