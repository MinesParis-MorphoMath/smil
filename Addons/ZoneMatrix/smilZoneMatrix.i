%include smilCommon.i

SMIL_MODULE(smilZoneMatrix)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DGrayLevelDistance.h"
%}

%import smilCore.i

%include "DGrayLevelDistance.h"

TEMPLATE_WRAP_FUNC(grayLevelZMSize);
TEMPLATE_WRAP_FUNC(grayLevelZMDistance);



