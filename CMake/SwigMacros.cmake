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

# SWIG WRAP CLASS
set(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS(class, name)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE} >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# Two template types, both variables
set(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_BOTH(class, name)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE},${_IMG_TYPE} >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# Two template types, first variable, second fixed
set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(class, fixedType, name)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(name \#\# _${_IMG_TYPE} \#\# _ \#\# fixedType) class<${_IMG_TYPE},fixedType >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# Two template types, first fixed, second variable
set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_FIX_FIRST(class, fixedType, name)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(name \#\# _ \#\# fixedType \#\# _${_IMG_TYPE}) class<fixedType,${_IMG_TYPE} >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS CLASS 2 TYPES
set(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_CROSS(class, name)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(name \#\# _${_IMG_TYPE} \#\#  _${_IMG_TYPE2}) class<${_IMG_TYPE},${_IMG_TYPE2}  >;\n"
    )
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS CLASS 3 TYPES
set(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_3T_CROSS(class, name)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    foreach(_IMG_TYPE3 ${IMAGE_TYPES})
      set(_STWD
          "${_STWD}  %template(name \#\# _${_IMG_TYPE} \#\#  _${_IMG_TYPE2} \#\#  _${_IMG_TYPE3}) class<${_IMG_TYPE},${_IMG_TYPE2},${_IMG_TYPE3}  >;\n"
      )
    endforeach(_IMG_TYPE3 ${IMAGE_TYPES})
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_MEMBER_FUNC(class, func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD "${_STWD}  %template(func) class::func<${_IMG_TYPE} >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS MEMBER FUNC 2 IMGS
set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_MEMBER_FUNC_2T_CROSS(class, func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(func) class::func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP FUNC
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC(func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD "${_STWD}  %template(func) smil::func<${_IMG_TYPE} >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP FUNC 2T (same types)
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T(func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE} >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# Two template types, first fixed, second variable
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_FIX_FIRST(func, fixedType)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD "${_STWD}  %template(func) smil::func<fixedType,${_IMG_TYPE} >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# Two template types, first variable, second fixed
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_FIX_SECOND(func, fixedType)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD "${_STWD}  %template(func) smil::func<${_IMG_TYPE},fixedType >;\n")
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 2 IMGS
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_CROSS(func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_CROSS(func)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    foreach(_IMG_TYPE3 ${IMAGE_TYPES})
      set(_STWD
          "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE2},${_IMG_TYPE3} >;\n"
      )
    endforeach(_IMG_TYPE3 ${IMAGE_TYPES})
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types, 2nd fixed
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_FIX_SECOND(func, fixedType)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(func) smil::func<${_IMG_TYPE},fixedType,${_IMG_TYPE2} >;\n"
    )
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types, last fixed
set(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_FIX_THIRD(func, fixedType)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE2},fixedType >;\n"
    )
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP SUBTYPES (ex. std::vector)
set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_SUBTYPE(class, subclass, basename)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(basename \#\# _ \#\# subclass \#\# _${_IMG_TYPE}) class< subclass<${_IMG_TYPE}> >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_SUBTYPES_CROSS(class, subclass, basename)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(basename \#\# _ \#\# subclass \#\# _${_IMG_TYPE} \#\# _${_IMG_TYPE2}) class< subclass <${_IMG_TYPE}, ${_IMG_TYPE2}> >;\n"
    )
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_SUBTYPES_FIX_FIRST(class, subclass, fixedType, basename)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(basename \#\# _ \#\# subclass \#\# _ \#\# fixedType \#\# _${_IMG_TYPE}) class< subclass <fixedType, ${_IMG_TYPE}> >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_SUBTYPES_FIX_SECOND(class, subclass, fixedType, basename)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(basename \#\# _ \#\# subclass \#\# _ \#\# _${_IMG_TYPE} \#\# fixedType) class< subclass <${_IMG_TYPE}, fixedType> >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP MAP
set(_STWD "${_STWD}%define TEMPLATE_WRAP_MAP_FIX_SECOND(class, name)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(name \#\# _${_IMG_TYPE}) map<${_IMG_TYPE}, class >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_MAP_CROSS_WITH_SECOND_SUBTYPE(class)\n")
foreach(_IMG_TYPE ${IMAGE_TYPES})
  foreach(_IMG_TYPE2 ${IMAGE_TYPES})
    set(_STWD
        "${_STWD}  %template(Map \#\# _${_IMG_TYPE}_ \#\# class \#\# _${_IMG_TYPE2}) map<${_IMG_TYPE}, class<${_IMG_TYPE2}> >;\n"
    )
  endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

set(_STWD
    "${_STWD}%define TEMPLATE_WRAP_MAP_CROSS_WITH_SECOND_SUBTYPE_FIX_FIRST(class, fixedType)\n"
)
foreach(_IMG_TYPE ${IMAGE_TYPES})
  set(_STWD
      "${_STWD}  %template(Map \#\# fixedType \#\# _ \#\# class \#\# _${_IMG_TYPE}) map<fixedType, class<${_IMG_TYPE}> >;\n"
  )
endforeach(_IMG_TYPE ${IMAGE_TYPES})
set(_STWD "${_STWD}%enddef\n\n")

# SUPPLEMENTARY DATA TYPES

if(IMAGE_TYPES_SUPPL)
  # SWIG WRAP SUPPL CLASS
  set(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_CLASS(class, name)\n")
  foreach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
    set(_STWD
        "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE} >;\n")
  endforeach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
  set(_STWD "${_STWD}%enddef\n\n")

  # SWIG WRAP SUPPL FUNC
  set(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC(func)\n")
  foreach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
    set(_STWD "${_STWD}  %template(func) smil::func<${_IMG_TYPE} >;\n")
  endforeach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
  set(_STWD "${_STWD}%enddef\n\n")

  # SWIG WRAP SUPPL CROSS FUNC 2 IMGS
  set(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(func)\n")
  foreach(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})
    foreach(_IMG_TYPE2 ${IMAGE_TYPES_SUPPL})
      set(_STWD
          "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE2} >;\n"
      )
    endforeach(_IMG_TYPE2 ${IMAGE_TYPES_SUPPL})
  endforeach(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})
  foreach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
    foreach(_IMG_TYPE2 ${IMAGE_TYPES})
      set(_STWD
          "${_STWD}  %template(func) smil::func<${_IMG_TYPE},${_IMG_TYPE2} >;\n"
      )
    endforeach(_IMG_TYPE2 ${IMAGE_TYPES})
  endforeach(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
  set(_STWD "${_STWD}%enddef\n\n")
else(IMAGE_TYPES_SUPPL)
  set(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_CLASS(class, name)\n%enddef\n")
  set(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC(func)\n%enddef\n")
  set(_STWD
      "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(func)\n%enddef\n")
endif(IMAGE_TYPES_SUPPL)

set(SWIG_TEMPLATE_WRAP_DEFINITIONS ${_STWD})

# SWIG_ADD_MODULE is deprecated from CMake 3.8
macro(ADD_SWIG_MODULE LIB_NAME LANGUAGE SOURCES)
  if(CMAKE_VERSION VERSION_LESS 3.8)
    swig_add_module(${LIB_NAME} ${LANGUAGE} ${SOURCES})
  else()
    swig_add_library(
      ${LIB_NAME}
      LANGUAGE ${LANGUAGE}
      SOURCES ${SOURCES})
  endif()
endmacro()
