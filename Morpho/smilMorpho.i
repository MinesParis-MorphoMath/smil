// Smil
// Copyright (c) 2010 Matthieu Faessel
//
// This file is part of Smil.
//
// Smil is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// Smil is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with Smil.  If not, see
// <http://www.gnu.org/licenses/>.



%{
/* Includes the header in the wrapper code */
#include "DTypes.hpp"
#include "DMorphoBase.hpp"
#include "DMorphoGeodesic.hpp"
#include "DMorphoExtrema.hpp"
%}
 

%extend StrElt
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}


%include "DTypes.hpp"
%include "DStructuringElement.h"
%include "DMorphoBase.hpp"
%include "DMorphoGeodesic.hpp"
%include "DMorphoExtrema.hpp"

//TEMPLATE_WRAP_FUNC(label);

TEMPLATE_WRAP_FUNC(dilate);
TEMPLATE_WRAP_FUNC(erode);
TEMPLATE_WRAP_FUNC(close);
TEMPLATE_WRAP_FUNC(open);
TEMPLATE_WRAP_FUNC(gradient);

TEMPLATE_WRAP_FUNC(label);

TEMPLATE_WRAP_FUNC(geoDil);
TEMPLATE_WRAP_FUNC(geoEro);
TEMPLATE_WRAP_FUNC(build);
TEMPLATE_WRAP_FUNC(dualBuild);
TEMPLATE_WRAP_FUNC(fillHoles);
TEMPLATE_WRAP_FUNC(levelPics);

TEMPLATE_WRAP_FUNC(hMinima);
TEMPLATE_WRAP_FUNC(hMaxima);
TEMPLATE_WRAP_FUNC(minima);
TEMPLATE_WRAP_FUNC(maxima);
