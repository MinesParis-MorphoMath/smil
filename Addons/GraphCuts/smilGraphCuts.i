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
%}

%import smilCore.i
%import smilMorpho.i

%include "Mosaic_GeoCuts.h"
// TEMPLATE_WRAP_FUNC(GeoCuts_MinSurfaces);
// TEMPLATE_WRAP_FUNC(GeoCuts_MinSurfaces_with_Line);
// TEMPLATE_WRAP_FUNC(GeoCuts_MinSurfaces_with_steps);
// TEMPLATE_WRAP_FUNC(GeoCuts_MinSurfaces_with_steps_vGradient);
// TEMPLATE_WRAP_FUNC(GeoCuts_MinSurfaces_with_steps_old);
TEMPLATE_WRAP_FUNC(GeoCuts_MultiWay_MinSurfaces);
// TEMPLATE_WRAP_FUNC(GeoCuts_Optimize_Mosaic);
// TEMPLATE_WRAP_FUNC(GeoCuts_Regularized_MinSurfaces);
// TEMPLATE_WRAP_FUNC(GeoCuts_Segment_Graph);
// TEMPLATE_WRAP_FUNC(MAP_MRF_edge_preserving);
// TEMPLATE_WRAP_FUNC(MAP_MRF_Ising);
// TEMPLATE_WRAP_FUNC(MAP_MRF_Potts);

TEMPLATE_WRAP_FUNC(testHandleSE);





