// Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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

SMIL_MODULE(smilMorpho)


%{
/* Includes the header in the wrapper code */
#include "DMorphoBase.hpp"
#include "DMorphoResidues.hpp"
#include "DMorphoGeodesic.hpp"
#include "DMorphoExtrema.hpp"
#include "DMorphoFilter.hpp"
#include "DMorphoArrow.hpp"
#include "DMorphoWatershed.hpp"
#include "DMorphoWatershedExtinction.hpp"
#include "DMorphoLabel.hpp"
#include "DCompositeSE.h"
#include "DHitOrMiss.hpp"
#include "DSkeleton.hpp"
#include "DMorphoInstance.h"
#include "DMorphoMaxTree.hpp"
#include "DMorphoGraph.hpp"
#include "DMorphoMeasures.hpp"
%}
 

// Import smilCore to have correct function signatures (arguments with Image_UINT8 instead of Image<unsigned char>)
%import smilCore.i


#ifdef SWIGPYTHON
%pythoncode %{

builtinOpen = open
%}
#endif // SWIGPYTHON

//////////////////////////////////////////////////////////
// Morpho Instance
//////////////////////////////////////////////////////////

%include "Core/include/private/DInstance.hpp"
%template(MorphoInstance) smil::UniqueInstance<smil::Morpho>;
%include "DMorphoInstance.h"


#ifdef SWIGJAVA
%ignore StrElt::operator ();
#endif // SWIGJAVA

#ifdef SWIGPYTHON
%include "DStructuringElement.h"
#endif // SWIGPYTHON


%include "DMorphoBase.hpp"
TEMPLATE_WRAP_FUNC(dilate);
TEMPLATE_WRAP_FUNC(erode);
TEMPLATE_WRAP_FUNC(close);
TEMPLATE_WRAP_FUNC(open);

%include "DMorphoResidues.hpp"
TEMPLATE_WRAP_FUNC(gradient);
TEMPLATE_WRAP_FUNC(topHat);
TEMPLATE_WRAP_FUNC(dualTopHat);

%include "DMorphoGeodesic.hpp"
TEMPLATE_WRAP_FUNC(geoDil);
TEMPLATE_WRAP_FUNC(geoEro);
TEMPLATE_WRAP_FUNC(geoBuild);
TEMPLATE_WRAP_FUNC(geoDualBuild);
TEMPLATE_WRAP_FUNC(build);
TEMPLATE_WRAP_FUNC(binBuild);
TEMPLATE_WRAP_FUNC(dualBuild);
TEMPLATE_WRAP_FUNC(hBuild);
TEMPLATE_WRAP_FUNC(hDualBuild);
TEMPLATE_WRAP_FUNC(buildOpen);
TEMPLATE_WRAP_FUNC(buildClose);
TEMPLATE_WRAP_FUNC(fillHoles);
TEMPLATE_WRAP_FUNC(levelPics);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist);
TEMPLATE_WRAP_FUNC_2T_CROSS(dist_euclidean);

%include "DMorphoExtrema.hpp"
TEMPLATE_WRAP_FUNC(hMinima);
TEMPLATE_WRAP_FUNC(hMaxima);
TEMPLATE_WRAP_FUNC(minima);
TEMPLATE_WRAP_FUNC(maxima);
TEMPLATE_WRAP_FUNC_2T_CROSS(minimaLabeled);
TEMPLATE_WRAP_FUNC_2T_CROSS(maximaLabeled);
TEMPLATE_WRAP_FUNC_2T_CROSS(hMinimaLabeled);
TEMPLATE_WRAP_FUNC_2T_CROSS(hMaximaLabeled);

%include "DMorphoFilter.hpp"
TEMPLATE_WRAP_FUNC(asfClose);
TEMPLATE_WRAP_FUNC(asfOpen);
TEMPLATE_WRAP_FUNC(mean);
TEMPLATE_WRAP_FUNC(median);

%include "DMorphoArrow.hpp"
TEMPLATE_WRAP_FUNC(arrow);

%include "DMorphoWatershed.hpp"
TEMPLATE_WRAP_FUNC(watershed);
TEMPLATE_WRAP_FUNC_2T_CROSS(watershed);
TEMPLATE_WRAP_FUNC_2T_CROSS(basins);
TEMPLATE_WRAP_FUNC(lblSkiz);
TEMPLATE_WRAP_FUNC_2T_CROSS(inflBasins);
TEMPLATE_WRAP_FUNC(inflZones);
TEMPLATE_WRAP_FUNC(waterfall);

%include "DMorphoWatershedExtinction.hpp"
%feature("director") ExtinctionFlooding;
TEMPLATE_WRAP_CLASS_MEMBER_FUNC(ExtinctionFlooding, floodWithExtValues);
TEMPLATE_WRAP_CLASS_MEMBER_FUNC(ExtinctionFlooding, floodWithExtRank);
TEMPLATE_WRAP_CLASS_2T_CROSS(ExtinctionFlooding, ExtinctionFlooding);
TEMPLATE_WRAP_FUNC_2T_CROSS(watershedExtinction);
TEMPLATE_WRAP_FUNC_3T_CROSS(watershedExtinction);
TEMPLATE_WRAP_FUNC_2T_CROSS(watershedExtinctionGraph);
TEMPLATE_WRAP_FUNC_3T_CROSS(watershedExtinctionGraph);

%include "DMorphoLabel.hpp"
TEMPLATE_WRAP_FUNC_2T_CROSS(label);
TEMPLATE_WRAP_FUNC_2T_CROSS(lambdaLabel);
TEMPLATE_WRAP_FUNC_2T_CROSS(fastLabel);
TEMPLATE_WRAP_FUNC_2T_CROSS(labelWithArea);
TEMPLATE_WRAP_FUNC_2T_CROSS(neighbors);

%ignore smil::CompStrEltList::operator[];
%extend smil::CompStrEltList
{
    CompStrElt &__getitem__(UINT n)
    {
	return self->compSeList[n];
    }
}

%include "DCompositeSE.h"
%include "DHitOrMiss.hpp"
TEMPLATE_WRAP_FUNC(hitOrMiss);
TEMPLATE_WRAP_FUNC(thin);
TEMPLATE_WRAP_FUNC(fullThin);
TEMPLATE_WRAP_FUNC(thick);
TEMPLATE_WRAP_FUNC(fullThick);

%include "DSkeleton.hpp"
TEMPLATE_WRAP_FUNC(skiz);
TEMPLATE_WRAP_FUNC(skeleton);
TEMPLATE_WRAP_FUNC_2T_CROSS(extinctionValues);
TEMPLATE_WRAP_FUNC(zhangSkeleton);

%include "DMorphoMaxTree.hpp"
TEMPLATE_WRAP_FUNC_2T_CROSS(ultimateOpen);
TEMPLATE_WRAP_FUNC_2T_CROSS(ultimateOpenMSER);
TEMPLATE_WRAP_FUNC(areaOpen);
TEMPLATE_WRAP_FUNC(areaClose);

%include "DMorphoGraph.hpp"
TEMPLATE_WRAP_FUNC_2T_CROSS(mosaicToGraph);
TEMPLATE_WRAP_FUNC(graphToMosaic);
TEMPLATE_WRAP_FUNC_2T_CROSS(drawGraph);

%include "DMorphoMeasures.hpp"
TEMPLATE_WRAP_FUNC(measGranulometry);


#ifdef SWIGPYTHON
%pythoncode %{

def open(*args):
    if type(args[0])==str:
      return builtinOpen(*args)
    else:
      return _smilMorphoPython.open(*args)
      
open.__doc__ = "Builtin function:\n" + builtinOpen.__doc__ + "\n\nSmil function:\n" + _smilMorphoPython.open.__doc__

%}
#endif // SWIGPYTHON
