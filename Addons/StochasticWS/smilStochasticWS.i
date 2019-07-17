%include smilCommon.i

SMIL_MODULE(smilStochasticWS)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
/* #include "DStochasticWatershed.hpp" */
#include "DStochasticWS.h"
%}

#include <jansson.hpp>

%import smilCore.i

// %include "DStochasticWatershed.hpp"
%include "DStochasticWS.h"

TEMPLATE_WRAP_FUNC_2T_CROSS(stochasticWatershed);
TEMPLATE_WRAP_FUNC_2T_CROSS(stochasticWatershedParallel);
TEMPLATE_WRAP_FUNC_2T_CROSS(stochasticFlatZones);
TEMPLATE_WRAP_FUNC_2T_CROSS(stochasticFlatZonesParallel);
TEMPLATE_WRAP_FUNC_2T_CROSS(overSegmentationCorrection);

