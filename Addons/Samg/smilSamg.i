%include smilCommon.i

SMIL_MODULE(smilSamg)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DSamgApplyThreshold.hpp"
#include "DSamgMeasures.hpp"
#include "DSamgArrow.hpp"
%}

%import smilCore.i

%include "DSamgApplyThreshold.hpp"

TEMPLATE_WRAP_FUNC(applyThreshold);
TEMPLATE_WRAP_FUNC(rangeThreshold);

TEMPLATE_WRAP_FUNC(rasterLabels);

// TEMPLATE_WRAP_FUNC(pruneSKIZ);

RES_T findTriplePoints (const Image<UINT16> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);
RES_T findTriplePoints (const Image<UINT16> &_im_, const Image<UINT16> &_skiz_, Image<UINT16> &_out_, const UINT& val, const StrElt &_se_);
// RES_T findTriplePoints (const Image<UINT8> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);

RES_T extendTriplePoints (Image<UINT8> &_triple_, const Image<UINT8> &_skiz_, const StrElt& _se_);
RES_T extendTriplePoints (Image<UINT16> &_triple_, const Image<UINT16> &_skiz_, const StrElt& _se_);
// RES_T extendTriplePoints (Image<UINT32> &_triple_, const Image<UINT32> &_skiz_, const StrElt& _se_);

TEMPLATE_WRAP_FUNC_2T_CROSS(dist_per_label);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist_cross_3d_per_label);

%include "DSamgMeasures.hpp"
TEMPLATE_WRAP_FUNC(measCrossCorrelation);
TEMPLATE_WRAP_FUNC(measHaralickFeatures);

%include "DSamgArrow.hpp"
TEMPLATE_WRAP_FUNC(hammingWeight)
TEMPLATE_WRAP_FUNC_2T_CROSS(arrowDual);
