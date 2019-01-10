%include smilCommon.i

SMIL_MODULE(smil3DBilateralFilter)

%import smilMorpho.i

%{
#include "DFilter.hpp"
%}

%import smilCore.i

%include "DFilter.hpp"

RES_T recursiveBilateralFilter (const Image<UINT8> &imIn, Image<UINT8> &imOut, float sigmaW, float sigmaR);
RES_T recursiveBilateralFilter (const Image<UINT16> &imIn, Image<UINT16> &imOut, float sigmaW, float sigmaR);

