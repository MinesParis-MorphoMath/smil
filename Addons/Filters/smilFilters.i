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

SMIL_MODULE(smilFilters)

%import smilCore.i

%{
/* Includes needed header(s)/definitions in the wrapped code */

#include "DfilterGabor.h"
#include "DfilterCanny.h"
#include "DfilterFastBilateral.h"
#include "DfilterDeriche.h"
#include "DfilterKuwahara.h"
#include "DfilterSigma.h"
#include "DfilterMeanShift.h"

#include "DfilterNormalize.h"
#include "DfilterGaussian.h"

#include "Dfilter3DBilateral.h"

#include "Dfilter3DBilateral.h"
%}

%import smilCore.i
# %import smilMorpho.i


// *******************************
// Filters from Morph-M
// *******************************
%include "DfilterGabor.h"
TEMPLATE_WRAP_FUNC(ImGaborFilterConvolution)
TEMPLATE_WRAP_FUNC(ImGaborFilterConvolutionNorm)
TEMPLATE_WRAP_FUNC(ImGaborFilterConvolutionNormAuto)

%include "DfilterCanny.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(ImCannyEdgeDetection)

%include "DfilterFastBilateral.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(ImFastBilateralFilter);

%include "DfilterDeriche.h"
TEMPLATE_WRAP_FUNC(ImDericheEdgeDetection);

%include "DfilterKuwahara.h"
TEMPLATE_WRAP_FUNC(ImKuwaharaFilter);
// TEMPLATE_WRAP_FUNC(ImKuwaharaFilterRGB);

%include "DfilterSigma.h"
TEMPLATE_WRAP_FUNC(ImSigmaFilter);
// TEMPLATE_WRAP_FUNC(ImSigmaFilterRGB);

%include "DfilterMeanShift.h"
TEMPLATE_WRAP_FUNC(ImMeanShiftFilter);
// TEMPLATE_WRAP_FUNC(ImMeanShiftFilterRGB);

// *******************************
// Filters from Jose-Marcio
// *******************************
%include "DfilterNormalize.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(ImNormalize);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImNormalizeAuto);
TEMPLATE_WRAP_FUNC_2T_CROSS(ImNormalizeSCurve);

%include "DfilterGaussian.h"
TEMPLATE_WRAP_FUNC(ImGaussianFilter);

// *******************************
// Filters from Theodore Chabardes
// *******************************
%include "Dfilter3DBilateral.h"
TEMPLATE_WRAP_FUNC(recursiveBilateralFilter);


