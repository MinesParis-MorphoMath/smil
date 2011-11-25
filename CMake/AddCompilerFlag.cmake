# Smil
# Copyright (c) 2010 Matthieu Faessel
#
# This file is part of Smil.
#
# Smil is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Smil is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Smil.  If not, see
# <http://www.gnu.org/licenses/>.


include (CheckCCompilerFlag)
include (CheckCXXCompilerFlag)
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
