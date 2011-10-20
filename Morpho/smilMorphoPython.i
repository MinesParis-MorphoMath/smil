%module smilMorphoPython

%include "../smilCommon.i"


%{
/* Includes the header in the wrapper code */
#include "DMorphoBase.hpp"
#include "DMorphoGeodesic.hpp"
#include "D_Types.h"
%}
 

%extend StrElt
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}


%include "D_Types.h"
%include "DStructuringElement.h"
%include "DMorphoBase.hpp"
%include "DMorphoGeodesic.hpp"

//TEMPLATE_WRAP_FUNC(label);

TEMPLATE_WRAP_FUNC(dilate);
TEMPLATE_WRAP_FUNC(erode);
TEMPLATE_WRAP_FUNC(close);
TEMPLATE_WRAP_FUNC(open);
TEMPLATE_WRAP_FUNC(gradient);

TEMPLATE_WRAP_FUNC(geoDil);
TEMPLATE_WRAP_FUNC(geoEro);
TEMPLATE_WRAP_FUNC(build);
TEMPLATE_WRAP_FUNC(dualBuild);

