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


# SWIG WRAP CLASS
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS(class, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# Two template types, both variables
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_BOTH(class, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE},${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# Two template types, first variable, second fixed
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(class, fixedType, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE} \#\# _ \#\# fixedType) class<${_IMG_TYPE},fixedType >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# Two template types, first fixed, second variable
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_FIX_FIRST(class, fixedType, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(name \#\# _ \#\# fixedType \#\# _${_IMG_TYPE}) class<fixedType,${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS CLASS 2 TYPES
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_2T_CROSS(class, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE}_${_IMG_TYPE2}) class<${_IMG_TYPE},${_IMG_TYPE2}  >;\n")
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_MEMBER_FUNC(class, func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(func) class::func<${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS MEMBER FUNC 2 IMGS
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_CLASS_MEMBER_FUNC_2T_CROSS(class, func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		SET(_STWD "${_STWD}  %template(func) class::func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")



# SWIG WRAP FUNC
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC(func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP FUNC 2T (same types)
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T(func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# Two template types, first fixed, second variable
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_FIX_FIRST(func, fixedType)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(func) func<fixedType,${_IMG_TYPE} >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# Two template types, first variable, second fixed
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_FIX_SECOND(func, fixedType)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},fixedType >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 2 IMGS
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_2T_CROSS(func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_CROSS(func)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		FOREACH(_IMG_TYPE3 ${IMAGE_TYPES})
			SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE2},${_IMG_TYPE3} >;\n")
		ENDFOREACH(_IMG_TYPE3 ${IMAGE_TYPES})
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types, 2nd fixed
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_FIX_SECOND(func, fixedType)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},fixedType,${_IMG_TYPE2} >;\n")
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")

# SWIG WRAP CROSS FUNC 3 Types, last fixed
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_FUNC_3T_FIX_THIRD(func, fixedType)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
		SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE2},fixedType >;\n")
	ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")


# SWIG WRAP VECTOR
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_VECTOR_SUBTYPE(class)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
        SET(_STWD "${_STWD}  %template(Vector_ \#\# class \#\# _${_IMG_TYPE}) vector< class<${_IMG_TYPE}> >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")


# SWIG WRAP MAP
SET(_STWD "${_STWD}%define TEMPLATE_WRAP_MAP_FIX_SECOND(class, name)\n")
FOREACH(_IMG_TYPE ${IMAGE_TYPES})
	SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE}) map<${_IMG_TYPE}, class >;\n")
ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES})
SET(_STWD "${_STWD}%enddef\n\n")


# SUPPLEMENTARY DATA TYPES

IF(IMAGE_TYPES_SUPPL)
	# SWIG WRAP SUPPL CLASS
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_CLASS(class, name)\n")
	FOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
		SET(_STWD "${_STWD}  %template(name \#\# _${_IMG_TYPE}) class<${_IMG_TYPE} >;\n")
	ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
	SET(_STWD "${_STWD}%enddef\n\n")

	# SWIG WRAP SUPPL FUNC
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC(func)\n")
	FOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
		SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE} >;\n")
	ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
	SET(_STWD "${_STWD}%enddef\n\n")


	# SWIG WRAP SUPPL CROSS FUNC 2 IMGS
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(func)\n")
	FOREACH(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})
		FOREACH(_IMG_TYPE2 ${IMAGE_TYPES_SUPPL})
			SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
		ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES_SUPPL})
	ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES} ${IMAGE_TYPES_SUPPL})
	FOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
		FOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
			SET(_STWD "${_STWD}  %template(func) func<${_IMG_TYPE},${_IMG_TYPE2} >;\n")
		ENDFOREACH(_IMG_TYPE2 ${IMAGE_TYPES})
	ENDFOREACH(_IMG_TYPE ${IMAGE_TYPES_SUPPL})
	SET(_STWD "${_STWD}%enddef\n\n")
ELSE(IMAGE_TYPES_SUPPL)
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_CLASS(class, name)\n%enddef\n")
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC(func)\n%enddef\n")
	SET(_STWD "${_STWD}%define TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(func)\n%enddef\n")
ENDIF(IMAGE_TYPES_SUPPL)



SET(SWIG_TEMPLATE_WRAP_DEFINITIONS ${_STWD})
