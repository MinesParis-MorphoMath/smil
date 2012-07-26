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




%include smilCommon.i

SMIL_MODULE(smilCore)


//////////////////////////////////////////////////////////
// Init
//////////////////////////////////////////////////////////

///// Numpy /////
#if defined SWIGPYTHON && defined USE_NUMPY
%init
%{
    // Required by NumPy in Python initialization
    import_array();
%}
#endif // defined SWIGPYTHON && defined USE_NUMPY


//////////////////////////////////////////////////////////
// Types
//////////////////////////////////////////////////////////

%{
/* Includes the header in the wrapper code */
#include "DTypes.hpp"
%}

%include "carrays.i"
%array_class(double, dArray);
//%array_class(void, voidArray);
%array_class(UINT8, uint8Array);

// BitArray
#ifdef SMIL_WRAP_Bit
%extend BitArray
{
//	std::string  __str__() {
//	    std::stringstream os;
//	    os << *self;
//	    return os.str();
//	}

//	bool operator[] (UINT i)
//	{
//	}

}
%ignore BitArray::operator++;
%include "DBitArray.h"
#endif // SMIL_WRAP_Bit

%include "DTypes.hpp"


//////////////////////////////////////////////////////////
// Typemaps
//////////////////////////////////////////////////////////

%rename(Index) operator[](UINT i);

PTR_ARG_OUT_APPLY(ret_min)
PTR_ARG_OUT_APPLY(ret_max)
PTR_ARG_OUT_APPLY(w)
PTR_ARG_OUT_APPLY(h)
PTR_ARG_OUT_APPLY(d)



//////////////////////////////////////////////////////////
// baseObject
//////////////////////////////////////////////////////////

%{
#include "DBaseObject.h"
%}

%extend baseObject 
{
	std::string  __str__() 
	{
	    std::stringstream os;
	    self->printSelf(os);
	    return os.str();
	}
}

%include "DBaseObject.h"


//////////////////////////////////////////////////////////
// Vectors
//////////////////////////////////////////////////////////

%include std_vector.i

%template(objVector) vector<baseObject*>;
%template(uintVector) vector<UINT>;




//////////////////////////////////////////////////////////
// Signals/Slots
//////////////////////////////////////////////////////////

#ifndef SWIGJAVA
// generate directors for Signal and Slot (for virtual methods overriding)
%feature("director") Signal;
%feature("director") Slot;
%feature("director") baseImageSlot;
#endif // SWIGJAVA

%include "DSignal.h"
%include "DSlot.h"
%include "DCoreEvents.h"


%template(baseImageSlot) Slot<baseImageEvent>;
%template(baseSlot) Slot<Event>;
%template(viewerFunctionSlot) MemberFunctionSlot<baseImageViewer, Event>;


//////////////////////////////////////////////////////////
// Image
//////////////////////////////////////////////////////////

%{
/* Includes the header in the wrapper code */
#include "DImage.hpp"
#include "DImage.hxx"
%}

// Import smilGui for viewers stuff
%import smilGui.i

%include "DBaseImage.h"
%include "DImage.hpp"

TEMPLATE_WRAP_CLASS(Image);
TEMPLATE_WRAP_FUNC(createImage);


//////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////

%{
/* Includes the header in the wrapper code */
#include "DImageArith.hpp"
#include "DImageDraw.hpp"
#include "DLineHistogram.hpp"
#include "DImageHistogram.hpp"
#include "DImageTransform.hpp"
%}


%include "DImageArith.hpp"
%include "DImageDraw.hpp"
%include "DLineHistogram.hpp"
%include "DImageHistogram.hpp"
%include "DImageTransform.hpp"


TEMPLATE_WRAP_FUNC_CROSS2(copy);

TEMPLATE_WRAP_FUNC(inv);
TEMPLATE_WRAP_FUNC(fill);
TEMPLATE_WRAP_FUNC(add);
TEMPLATE_WRAP_FUNC(addNoSat);
TEMPLATE_WRAP_FUNC(sub);
TEMPLATE_WRAP_FUNC(subNoSat);
TEMPLATE_WRAP_FUNC(mul);
TEMPLATE_WRAP_FUNC(mulNoSat);
TEMPLATE_WRAP_FUNC(div);

TEMPLATE_WRAP_FUNC(equ);
TEMPLATE_WRAP_FUNC(diff);
TEMPLATE_WRAP_FUNC(sup);
TEMPLATE_WRAP_FUNC(inf);
TEMPLATE_WRAP_FUNC(low);
TEMPLATE_WRAP_FUNC(lowOrEqu);
TEMPLATE_WRAP_FUNC(grt);
TEMPLATE_WRAP_FUNC(grtOrEqu);
TEMPLATE_WRAP_FUNC(logicAnd);
TEMPLATE_WRAP_FUNC(logicOr);
TEMPLATE_WRAP_FUNC(test);

TEMPLATE_WRAP_FUNC(vol);
TEMPLATE_WRAP_FUNC(minVal);
TEMPLATE_WRAP_FUNC(maxVal);
TEMPLATE_WRAP_FUNC(rangeVal);


TEMPLATE_WRAP_FUNC(histo);

TEMPLATE_WRAP_FUNC(thresh);
TEMPLATE_WRAP_FUNC(stretchHist);
TEMPLATE_WRAP_FUNC(enhanceContrast);

TEMPLATE_WRAP_FUNC(drawRectangle);


TEMPLATE_WRAP_FUNC(vFlip);
TEMPLATE_WRAP_FUNC(trans);
TEMPLATE_WRAP_FUNC(resize);




