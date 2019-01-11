%include smilCommon.i

SMIL_MODULE(smilChabardes)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DApplyThreshold.hpp"
#include "DMeasures.hpp"
#include "DArrow.hpp"
%}

%import smilCore.i

%include "DApplyThreshold.hpp"
RES_T applyThreshold (const Image<UINT8> &_im_, const vector<UINT8>& modes, Image<UINT8>& _out_);
RES_T applyThreshold (const Image<UINT16> &_im_, const vector<UINT16>& modes, Image<UINT16>& _out_);
RES_T areaThreshold (const Image<UINT8> &_im_, const UINT8 &threshold, Image<UINT8> &_out_);
RES_T areaThreshold (const Image<UINT16> &_im_, const UINT16 &threshold, Image<UINT16> &_out_);
RES_T rangeThreshold (const Image<UINT8> &_im_, const UINT8 &threshold, Image<UINT8> &_out_);
RES_T rangeThreshold (const Image<UINT16> &_im_, const UINT16 &threshold, Image<UINT16> &_out_);
RES_T rasterLabels (const Image<UINT8> &_im_, Image<UINT8>& _out_);
RES_T rasterLabels (const Image<UINT16> &_im_, Image<UINT16>& _out_);
RES_T pruneSKIZ (const Image<UINT8> &_im_, Image<UINT8> &_out_, const StrElt &_se_);
RES_T pruneSKIZ (const Image<UINT16> &_im_, Image<UINT16> &_out_, const StrElt &_se_);
RES_T find_triple_points (const Image<UINT16> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);
RES_T find_triple_points (const Image<UINT16> &_im_, const Image<UINT16> &_skiz_, Image<UINT16> &_out_, const UINT& val, const StrElt &_se_);
RES_T find_triple_points (const Image<UINT8> &_im_, const Image<UINT8> &_skiz_, Image<UINT8> &_out_, const UINT& val, const StrElt &_se_);
RES_T extend_triple_points (Image<UINT8> &_triple_, const Image<UINT8> &_skiz_, const StrElt& _se_);
RES_T extend_triple_points (Image<UINT16> &_triple_, const Image<UINT16> &_skiz_, const StrElt& _se_);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist_per_label);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist_cross_3d_per_label);

%include "DMeasures.hpp"
vector<double> measCrossCorrelation(const Image<UINT8> &imIn, const UINT8 &val1, const UINT8 &val2, size_t dx, size_t dy, size_t dz, UINT maxSteps=0, bool normalize=false);
vector<double> measCrossCorrelation(const Image<UINT16> &imIn, const UINT16 &val1, const UINT16 &val2, size_t dx, size_t dy, size_t dz, UINT maxSteps=0, bool normalize=false);
vector<double> measHaralickFeatures (Image<UINT8> &imIn, const StrElt &s);
vector<double> measHaralickFeatures (Image<UINT16> &imIn, const StrElt &s);

%include "DArrow.hpp"
TEMPLATE_WRAP_FUNC (hammingWeight)
TEMPLATE_WRAP_FUNC_2T_CROSS(arrowDual);
