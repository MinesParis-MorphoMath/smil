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

SMIL_MODULE(smilGraphCuts)

%{
/* Includes needed header(s)/definitions in the wrapped code */
#include "Mosaic_GeoCuts.h"
#include "GeoCuts.h"
#include "GeoCuts_MinSurfaces.h"
#include "GeoCuts_Watershed.h"
#include "GeoCuts_Markov.h"

%}

%import smilCore.i
%import smilMorpho.i

%include "Mosaic_GeoCuts.h"
TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces);
// TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces_with_Line);
// TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces_with_steps);
// TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces_with_steps_vGradient);
// TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces_with_steps_old);
TEMPLATE_WRAP_FUNC(geoCutsMultiWay_MinSurfaces);
// TEMPLATE_WRAP_FUNC(geoCutsRegularized_MinSurfaces);
// TEMPLATE_WRAP_FUNC(geoCutsOptimize_Mosaic);
// TEMPLATE_WRAP_FUNC(geoCutsSegment_Graph);

TEMPLATE_WRAP_FUNC(testHandleSE);

//
// Sub module GeoCuts
//
%include "GeoCuts.h"
//
%include "GeoCuts_MinSurfaces.h"
//
// ++ in src : geoCuts
TEMPLATE_WRAP_FUNC_2T_CROSS(geoCuts);
// TEMPLATE_WRAP_FUNC(geoCutsBoundary_Constrained_MinSurfaces);
TEMPLATE_WRAP_FUNC_2T_CROSS(geoCutsMinSurfaces);
// TEMPLATE_WRAP_FUNC(geoCutsMinSurfaces_With_Line);
TEMPLATE_WRAP_FUNC_2T_CROSS(geoCutsMultiway_MinSurfaces);
// TEMPLATE_WRAP_FUNC(geoCutsStochastic_Watershed_Variance);
// TEMPLATE_WRAP_FUNC(geoCutsParametric);

//
%include "GeoCuts_Watershed.h"
//
// TEMPLATE_WRAP_FUNC(geoCutsBiCriteria_Shortest_Forest);
// TEMPLATE_WRAP_FUNC(geoCutsLexicographical_Shortest_Forest);
// TEMPLATE_WRAP_FUNC(geoCutsMax_Fiability_Forest);
TEMPLATE_WRAP_FUNC_2T_CROSS(geoCutsMultiway_Watershed);
// TEMPLATE_WRAP_FUNC(geoCutsReg_SpanningForest);
// TEMPLATE_WRAP_FUNC(geoCutsStochastic_Watershed);
// TEMPLATE_WRAP_FUNC(geoCutsStochastic_Watershed_2);
// TEMPLATE_WRAP_FUNC(geoCutsVectorial_Lexicographical_Shortest_Forest);
// TEMPLATE_WRAP_FUNC(geoCutsVectorial_Shortest_Forest);
TEMPLATE_WRAP_FUNC_2T_CROSS(geoCutsWatershed_MinCut);
// TEMPLATE_WRAP_FUNC(geoCutsWatershed_Prog_MinCut);
// TEMPLATE_WRAP_FUNC(geoCutsWatershed_SpanningForest);
// TEMPLATE_WRAP_FUNC(geoCutsWatershed_SpanningForest_v2);
// TEMPLATE_WRAP_FUNC(geoCutsWatershed_SPF);

//
%include "GeoCuts_Markov.h"
//
// TEMPLATE_WRAP_FUNC(MAP_MRF_edge_preserving);
// TEMPLATE_WRAP_FUNC(MAP_MRF_Ising);
// TEMPLATE_WRAP_FUNC(MAP_MRF_Potts);



