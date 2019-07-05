// Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



%include smilCommon.i

SMIL_MODULE(smilAdvanced)


%{
/* Includes needed header(s)/definitions in the wrapped code */
#include "DAdvanced.h"

%}

%import smilCore.i
%import smilMorpho.i


%include "DMorphoPathOpening.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(ImPathOpeningBruteForce);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImPathClosingBruteForce);
TEMPLATE_WRAP_FUNC(ImPathOpening);
TEMPLATE_WRAP_FUNC(ImPathClosing);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImUltimatePathOpening);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImUltimatePathClosing);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImBinaryPathOpening);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImBinaryPathClosing);

TEMPLATE_WRAP_FUNC(ImGeodesicPathOpening);
TEMPLATE_WRAP_FUNC(ImGeodesicPathClosing);
TEMPLATE_WRAP_FUNC(ImUltimateGeodesicPathOpening);
TEMPLATE_WRAP_FUNC(ImUltimateGeodesicPathClosing);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImGeodesicElongation);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImGeodesicExtremities);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImLabelFlatZonesWithElongation);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImLabelFlatZonesWithExtremities);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImLabelFlatZonesWithGeodesicDiameter);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImGeodesicDiameter);
//BMI TEMPLATE_WRAP_FUNC(ImGeodesicTortuosity);

TEMPLATE_WRAP_FUNC_2T_CROSS(ImUltimatePathOpening_GraphV2);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImUltimatePathClosing_GraphV2);
TEMPLATE_WRAP_FUNC(ImPathClosing_GraphV2);
TEMPLATE_WRAP_FUNC(ImPathOpening_GraphV2);

TEMPLATE_WRAP_FUNC_2T_CROSS(ImThresholdWithUniqueCCForBackGround);
TEMPLATE_WRAP_FUNC(CountNbCCperThreshold);
TEMPLATE_WRAP_FUNC(CountNbPixelOfNDG);
TEMPLATE_WRAP_FUNC(MeanValueOf);
TEMPLATE_WRAP_FUNC_2T_CROSS(PseudoPatternSpectrum);

TEMPLATE_WRAP_FUNC_2T_CROSS(ImSupSmallRegion);
TEMPLATE_WRAP_FUNC_2T_CROSS(measComputeVolume);
TEMPLATE_WRAP_FUNC_2T_CROSS(measComputeIndFromPatternSpectrum);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImThresholdWithMuAndSigma);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImElongationFromSkeleton);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImFromSkeletonSupTriplePoint);
TEMPLATE_WRAP_FUNC(FromSkeletonComputeGranulometry);

TEMPLATE_WRAP_FUNC_2T_CROSS(ImFromSK_AreaForEachCC);

### Has prototype but not implementation
### TEMPLATE_WRAP_FUNC_2T_CROSS(ImLayerDist);

//  -> check if necessary, otherwise remove it
//TEMPLATE_WRAP_FUNC(ImFalseColorHSL);
//  -> check the use of RGB
//TEMPLATE_WRAP_FUNC(GetConfusionMatrix);

%include "DGrayLevelDistance.h"

TEMPLATE_WRAP_FUNC(grayLevelSizeZM);
TEMPLATE_WRAP_FUNC(grayLevelDistanceZM);
TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Diameter);
TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Elongation);
TEMPLATE_WRAP_FUNC(grayLevelDistanceZM_Tortuosity);

//
//
//
%include "DFastAreaOpening.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaClosing_PixelQueue);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaOpening_PixelQueue);

//TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaOpening_MaxTree);
//TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaClosing_MaxTree);
//TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaOpening_UnionFind);
//TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaClosing_UnionFind);

TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaOpening_Line);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaClosing_Line);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImAreaOpening_LineSupEqu);

//TEMPLATE_WRAP_FUNC_2T_CROSS(ImInertiaThinning_MaxTree);
//TEMPLATE_WRAP_FUNC_2T_CROSS(ImInertiaThickening_MaxTree);

%include "DFastLine.h"
//TEMPLATE_WRAP_FUNC(ImLineOpen);
//TEMPLATE_WRAP_FUNC(ImLineClose);
//TEMPLATE_WRAP_FUNC(ImLineDilate);
//TEMPLATE_WRAP_FUNC(ImLineErode);

//TEMPLATE_WRAP_FUNC(ImSquareOpen);
//TEMPLATE_WRAP_FUNC(ImSquareClose);
//TEMPLATE_WRAP_FUNC(ImSquareDilate);
//TEMPLATE_WRAP_FUNC(ImSquareErode);

TEMPLATE_WRAP_FUNC(ImFastLineOpen_Morard);
//TEMPLATE_WRAP_FUNC(ImFastLineClose_Morard);

