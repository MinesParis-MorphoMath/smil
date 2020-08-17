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
// #include "DfilterFastBilateral.h"
#include "DfilterDeriche.h"
#include "DfilterKuwahara.h"
#include "DfilterSigma.h"
#include "DfilterMeanShift.h"

#include "DfilterScaleRange.h"
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
TEMPLATE_WRAP_FUNC(gaborFilterConvolution)
TEMPLATE_WRAP_FUNC(gaborFilterConvolutionNorm)
TEMPLATE_WRAP_FUNC(gaborFilterConvolutionNormAuto)

%include "DfilterCanny.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(cannyEdgeDetection)

// %include "DfilterFastBilateral.h"
// TEMPLATE_WRAP_FUNC_2T_CROSS(ImFastBilateralFilter);

%include "DfilterDeriche.h"
TEMPLATE_WRAP_FUNC(dericheEdgeDetection);

%include "DfilterKuwahara.h"
TEMPLATE_WRAP_FUNC(kuwaharaFilter);
// TEMPLATE_WRAP_FUNC(kuwaharaFilterRGB);

%include "DfilterSigma.h"
TEMPLATE_WRAP_FUNC(sigmaFilter);
// TEMPLATE_WRAP_FUNC(sigmaFilterRGB);

%include "DfilterMeanShift.h"
TEMPLATE_WRAP_FUNC(meanShiftFilter);
// TEMPLATE_WRAP_FUNC(meanShiftFilterRGB);

// *******************************
// Filters from Jose-Marcio
// *******************************
%include "DfilterScaleRange.h"
TEMPLATE_WRAP_FUNC_2T_CROSS(imageScaleRange);
// TEMPLATE_WRAP_FUNC_2T_CROSS(imageScaleRangeAuto);
TEMPLATE_WRAP_FUNC_2T_CROSS(imageScaleRangeSCurve);

%include "DfilterGaussian.h"
TEMPLATE_WRAP_FUNC(ImGaussianFilter);

// *******************************
// Filters from Theodore Chabardes
// *******************************
%include "Dfilter3DBilateral.h"
TEMPLATE_WRAP_FUNC(recursiveBilateralFilter);



