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


//  -> check if necessary, otherwise remove it
//TEMPLATE_WRAP_FUNC(ImFalseColorHSL);
//  -> check the use of RGB
//TEMPLATE_WRAP_FUNC(GetConfusionMatrix);


%include "DLineMorpho.h"
TEMPLATE_WRAP_FUNC(lineDilate);
TEMPLATE_WRAP_FUNC(lineErode);
TEMPLATE_WRAP_FUNC(lineOpen);
TEMPLATE_WRAP_FUNC(lineClose);

TEMPLATE_WRAP_FUNC(squareDilate);
TEMPLATE_WRAP_FUNC(squareErode);
TEMPLATE_WRAP_FUNC(squareOpen);
TEMPLATE_WRAP_FUNC(squareClose);

TEMPLATE_WRAP_FUNC(circleDilate);
TEMPLATE_WRAP_FUNC(circleErode);
TEMPLATE_WRAP_FUNC(circleOpen);
TEMPLATE_WRAP_FUNC(circleClose);

TEMPLATE_WRAP_FUNC(rectangleDilate);
TEMPLATE_WRAP_FUNC(rectangleErode);
TEMPLATE_WRAP_FUNC(rectangleOpen);
TEMPLATE_WRAP_FUNC(rectangleClose);
//
//TEMPLATE_WRAP_FUNC(imFastLineOpen);
//TEMPLATE_WRAP_FUNC(ImFastLineClose_Morard);



%include "private/AdvancedGeodesy/GeodesicPathOpening.hpp"

TEMPLATE_WRAP_FUNC_2T_CROSS(labelFlatZonesWithProperty);

TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicDiameter);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicElongation);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicTortuosity);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicExtremities);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicProperty);

TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicPathOpening);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicPathClosing);

TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicUltimatePathOpening);
TEMPLATE_WRAP_FUNC_2T_CROSS(geodesicUltimatePathClosing);


%include "DAreaOpen.h"
%include "private/AreaOpening/DAreaOpen.hpp"
%include "private/AreaOpening/DAreaOpenUnionFind.hpp"
TEMPLATE_WRAP_FUNC(areaOpening);
//TEMPLATE_WRAP_FUNC(areaClosing);
