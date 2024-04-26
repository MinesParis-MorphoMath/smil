%include smilCommon.i

SMIL_MODULE(smilSamg)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DApplyThreshold.hpp"
#include "DMeasures.hpp"
#include "DArrow.hpp"
%}

%import smilCore.i

%include "DApplyThreshold.hpp"

TEMPLATE_WRAP_FUNC(applyThreshold);

TEMPLATE_WRAP_FUNC(rangeThreshold);

TEMPLATE_WRAP_FUNC(rasterLabels);

// RES_T pruneSKIZ (const Image<UINT8> &_im_, Image<UINT8> &_out_, const StrElt &_se_);
// RES_T pruneSKIZ (const Image<UINT16> &_im_, Image<UINT16> &_out_, const StrElt &_se_);
// RES_T pruneSKIZ (const Image<UINT32> &_im_, Image<UINT32> &_out_, const StrElt &_se_);

RES_T findTriplePoints (const Image<UINT16> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);
RES_T findTriplePoints (const Image<UINT16> &_im_, const Image<UINT16> &_skiz_, Image<UINT16> &_out_, const UINT& val, const StrElt &_se_);
// RES_T findTriplePoints (const Image<UINT8> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);

RES_T extendTriplePoints (Image<UINT8> &_triple_, const Image<UINT8> &_skiz_, const StrElt& _se_);
RES_T extendTriplePoints (Image<UINT16> &_triple_, const Image<UINT16> &_skiz_, const StrElt& _se_);
// RES_T extendTriplePoints (Image<UINT32> &_triple_, const Image<UINT32> &_skiz_, const StrElt& _se_);

TEMPLATE_WRAP_FUNC_2T_CROSS(dist_per_label);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist_cross_3d_per_label);

%include "DMeasures.hpp"
TEMPLATE_WRAP_FUNC(measCrossCorrelation);

TEMPLATE_WRAP_FUNC(measHaralickFeatures);

//vector<double> measHaralickFeatures (Image<UINT8> &imIn, const StrElt &s);
//vector<double> measHaralickFeatures (Image<UINT16> &imIn, const StrElt &s);
//vector<double> measHaralickFeatures (Image<UINT32> &imIn, const StrElt &s);

%include "DArrow.hpp"
TEMPLATE_WRAP_FUNC(hammingWeight)
TEMPLATE_WRAP_FUNC_2T_CROSS(arrowDual);
