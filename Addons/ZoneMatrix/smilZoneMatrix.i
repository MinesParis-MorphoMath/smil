%include smilCommon.i

SMIL_MODULE(smilZoneMatrix)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DGrayLevelDistance.h"
%}

%import smilCore.i

%include "DGrayLevelDistance.h"

// TEMPLATE_WRAP_FUNC(grayLevelSizeZM);
// TEMPLATE_WRAP_FUNC(grayLevelDistanceZM);
// TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Diameter);
// TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Elongation);
// TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Tortuosity);



