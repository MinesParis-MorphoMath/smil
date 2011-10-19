%module smilMorphoPython

%feature("autodoc", "1");

%include <windows.i>

%{
/* Includes the header in the wrapper code */
#include "DImageMorph.hpp"

#include "DCommon.h"
#include "DImage.h"
#include "DImage.hpp"
#include "DImage.hxx"
#include "D_Types.h"
#include "D_BaseObject.h"
#include "DBaseImageOperations.hpp"
#include "DBaseLineOperations.hpp"
#include "DImageArith.hpp"
/*#include "D_BaseOperations.h"*/
#include "DImageIO_PNG.h"
#include "memory"
%}
 

%rename(__lshift__)  operator<<; 


%include "DStructuringElement.h"
%include "DImageMorph.hpp"

%define TEMPLATE_WRAP_FUNC_TYPE(func)
  %template(func) func<UINT8>;
/*  %template(func) func<UINT16>; */
/*  %template(func) func<UINT32>; */
%enddef

%extend StrElt
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}


//TEMPLATE_WRAP_FUNC_TYPE(label);

TEMPLATE_WRAP_FUNC_TYPE(dilate);
TEMPLATE_WRAP_FUNC_TYPE(erode);
TEMPLATE_WRAP_FUNC_TYPE(close);
TEMPLATE_WRAP_FUNC_TYPE(open);
TEMPLATE_WRAP_FUNC_TYPE(gradient);
TEMPLATE_WRAP_FUNC_TYPE(geoDil);
TEMPLATE_WRAP_FUNC_TYPE(geoEro);
TEMPLATE_WRAP_FUNC_TYPE(build);
TEMPLATE_WRAP_FUNC_TYPE(dualBuild);

