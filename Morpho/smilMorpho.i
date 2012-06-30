// Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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


#ifdef SWIGPYTHON
%module smilMorphoPython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module smilMorphoJava
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%module smilMorphoOctave
#endif // SWIGOCTAVE


%include smilCommon.i

%{
/* Includes the header in the wrapper code */
#include "DMorphoBase.hpp"
#include "DMorphoGeodesic.hpp"
#include "DMorphoExtrema.hpp"
#include "DMorphoArrow.hpp"
#include "DMorphoWatershed.hpp"
#include "DMorphoLabel.hpp"
%}
 

%extend StrElt
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}

#ifdef SWIGJAVA
%ignore StrElt::operator ();
#endif // SWIGJAVA


%include "DStructuringElement.h"
%include "DMorphoBase.hpp"
%include "DMorphoGeodesic.hpp"
%include "DMorphoExtrema.hpp"
%include "DMorphoArrow.hpp"
%include "DMorphoWatershed.hpp"

// Import smilCore to have correct function signatures (arguments with Image_UINT8 instead of Image<unsigned char>)
%import smilCore.i


TEMPLATE_WRAP_FUNC(dilate);
TEMPLATE_WRAP_FUNC(erode);
TEMPLATE_WRAP_FUNC(close);
TEMPLATE_WRAP_FUNC(open);
TEMPLATE_WRAP_FUNC(gradient);

TEMPLATE_WRAP_FUNC(geoDil);
TEMPLATE_WRAP_FUNC(geoEro);
TEMPLATE_WRAP_FUNC(geoBuild);
TEMPLATE_WRAP_FUNC(geoDualBuild);
TEMPLATE_WRAP_FUNC(build);
TEMPLATE_WRAP_FUNC(dualBuild);
TEMPLATE_WRAP_FUNC(fillHoles);
TEMPLATE_WRAP_FUNC(levelPics);
TEMPLATE_WRAP_FUNC(dist);

TEMPLATE_WRAP_FUNC(hMinima);
TEMPLATE_WRAP_FUNC(hMaxima);
TEMPLATE_WRAP_FUNC(minima);
TEMPLATE_WRAP_FUNC(maxima);

TEMPLATE_WRAP_FUNC(arrow);
TEMPLATE_WRAP_FUNC(arrowGrt);
TEMPLATE_WRAP_FUNC(arrowGrtOrEqu);
TEMPLATE_WRAP_FUNC(arrowEqu);

TEMPLATE_WRAP_FUNC(watershed);
TEMPLATE_WRAP_FUNC_CROSS2(watershed);


%include "DMorphoLabel.hpp"

TEMPLATE_WRAP_FUNC_CROSS2(label);
