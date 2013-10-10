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
// Errors
//////////////////////////////////////////////////////////

%ignore Error;
class exception{};
%include "DErrors.h"



//////////////////////////////////////////////////////////
// Types
//////////////////////////////////////////////////////////

%include "carrays.i"
//%array_class(double, DArray);
//%array_class(void, VoidArray);
//%array_class(UINT8, Uint8Array);


%include "DTypes.hpp"
%include "DTypes.h"


//////////////////////////////////////////////////////////
// Typemaps
//////////////////////////////////////////////////////////

%rename(Index) operator[](UINT i);

PTR_ARG_OUT_APPLY(ret_min)
PTR_ARG_OUT_APPLY(ret_max)
PTR_ARG_OUT_APPLY(w)
PTR_ARG_OUT_APPLY(h)
PTR_ARG_OUT_APPLY(d)
PTR_ARG_OUT_APPLY(s)



//////////////////////////////////////////////////////////
// BaseObject
//////////////////////////////////////////////////////////

%extend smil::BaseObject 
{
	std::string  __str__() 
	{
	    std::stringstream os;
	    self->printSelf(os);
	    return os.str();
	}
}


%include "DCommon.h"
%include "DBaseObject.h"



//////////////////////////////////////////////////////////
// Vectors
//////////////////////////////////////////////////////////

#ifndef SWIGXML


%include std_vector.i

// Expose std::vector<> as a Python list
namespace std 
{
    %template(UintVector) vector<UINT>;
    %template(UcharVector) vector<UINT8>;
    %template(UshortVector) vector<UINT16>;
    %template(DoubleVector) vector<double>;
    %template(StringVector) vector<string>;
    
    %template(DoubleMatrix) vector<DoubleVector>;
}

#endif // SWIGXML

//////////////////////////////////////////////////////////
// Maps
//////////////////////////////////////////////////////////

#ifndef SWIGXML


%include std_map.i


// Expose std::map<> as a Python dict
namespace std 
{
    %template(UintDoubleMap) map<UINT,double>;
    %template(UintDoubleVectorMap) map<UINT,DoubleVector>;
    %template(UintUintVectorMap) map<UINT,UintVector>;
    
    TEMPLATE_WRAP_CLASS_2T_BOTH(map, Map)
    
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, UINT, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, double, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, RGB, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, IntPoint, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, DoublePoint, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, DoubleVector, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, UintVector, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, Box, Map)
}

#endif // SWIGXML

//////////////////////////////////////////////////////////
// Core Instance
//////////////////////////////////////////////////////////

%include "DInstance.hpp"
%template(CoreInstance) smil::UniqueInstance<Core>;
%include "DCoreInstance.h"

#ifndef SWIGXML

namespace std 
{
    %template(ObjVector) vector<BaseObject*>;
}

#endif // SWIGXML

//////////////////////////////////////////////////////////
// Signals/Slots
//////////////////////////////////////////////////////////

#ifndef SWIGJAVA
// generate directors for Signal and Slot (for virtual methods overriding)
%feature("director") Signal;
%feature("director") Slot;
%feature("director") BaseImageSlot;
#endif // SWIGJAVA

%include "DSignal.h"
%include "DSlot.h"
%include "DCoreEvents.h"


namespace smil
{
    %template(BaseImageSlot) Slot<BaseImageEvent>;
    %template(EventSlot) Slot<Event>;
    %template(ViewerFunctionSlot) MemberFunctionSlot<BaseImageViewer, Event>;
    %template(FunctionSlot_) FunctionSlot<Event>;
}

//////////////////////////////////////////////////////////
// Image
//////////////////////////////////////////////////////////

#ifndef SWIGIMPORTED
%include "NSTypes.i"
#endif

// Import smilGui for viewers stuff
%import smilGui.i

%ignore smil::Image::operator[];
%extend smil::Image
{
    T __getitem__(size_t i) { return self->getPixel(i); }
    RES_T __setitem__(size_t i, T val) { return self->setPixel(i, val); }
}

%include "DBaseImage.h"
%include "DImage.hpp"
%include "DSharedImage.hpp"

#ifndef SWIGXML

namespace std 
{
    %template(ImgVector) std::vector<BaseImage*>;
}

#endif // SWIGXML

namespace smil
{
    TEMPLATE_WRAP_CLASS(Image, Image);
    TEMPLATE_WRAP_FUNC(createImage);
    TEMPLATE_WRAP_CLASS(SharedImage, SharedImage);
}

