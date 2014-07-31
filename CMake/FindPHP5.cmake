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


set(PHP5_POSSIBLE_INCLUDE_PATHS
  /usr/include/php5
  /usr/local/include/php5
  /usr/include/php
  /usr/local/include/php
  /usr/local/apache/php
  )

set(PHP5_POSSIBLE_LIB_PATHS
  /usr/lib
  )

find_path(PHP5_FOUND_INCLUDE_PATH main/php.h
  ${PHP5_POSSIBLE_INCLUDE_PATHS})

if(PHP5_FOUND_INCLUDE_PATH)
  set(php5_paths "${PHP5_POSSIBLE_INCLUDE_PATHS}")
  foreach(php5_path Zend main TSRM)
    set(php5_paths ${php5_paths} "${PHP5_FOUND_INCLUDE_PATH}/${php5_path}")
  endforeach()
  set(PHP5_INCLUDE_PATH "${php5_paths}" INTERNAL "PHP5 include paths")
endif()

find_program(PHP5_EXECUTABLE NAMES php5 php )

mark_as_advanced(
  PHP5_EXECUTABLE
  PHP5_FOUND_INCLUDE_PATH
  )

if(APPLE)
# this is a hack for now
  set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS
   "${CMAKE_SHARED_MODULE_CREATE_C_FLAGS} -Wl,-flat_namespace")
  foreach(symbol
    __efree
    __emalloc
    __estrdup
    __object_init_ex
    __zend_get_parameters_array_ex
    __zend_list_find
    __zval_copy_ctor
    _add_property_zval_ex
    _alloc_globals
    _compiler_globals
    _convert_to_double
    _convert_to_long
    _zend_error
    _zend_hash_find
    _zend_register_internal_class_ex
    _zend_register_list_destructors_ex
    _zend_register_resource
    _zend_rsrc_list_get_rsrc_type
    _zend_wrong_param_count
    _zval_used_for_init
    )
    set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS
      "${CMAKE_SHARED_MODULE_CREATE_C_FLAGS},-U,${symbol}")
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PHP5 DEFAULT_MSG PHP5_EXECUTABLE PHP5_INCLUDE_PATH)
