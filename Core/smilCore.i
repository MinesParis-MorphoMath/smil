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


%{
#include "DImage.hxx"
%}


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
%include "DErrors.h"



//////////////////////////////////////////////////////////
// Types
//////////////////////////////////////////////////////////


%include "DTypes.hpp"



// BitArray
#ifdef SMIL_WRAP_BIT
%include "Bit.i"
#endif // SMIL_WRAP_BIT

// RGB
#ifdef SMIL_WRAP_RGB
%include "RGB.i"
#else
%include "DColor.h"
#endif // SMIL_WRAP_RGB


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

%template(DoublePoint) Point<double>;
%template(IntPoint) Point<int>;


//////////////////////////////////////////////////////////
// Vectors
//////////////////////////////////////////////////////////

#ifndef SWIGXML


%include std_vector.i

// Expose std::vector<> as a Python list
namespace std 
{
    %template(Vector_UINT) vector<UINT>;
    %template(Vector_UINT8) vector<UINT8>;
    %template(Vector_UINT16) vector<UINT16>;
    %template(Vector_int) vector<int>;
    %template(Vector_double) vector<double>;
    %template(Vector_string) vector<string>;
    
    %template(Matrix_double) vector<Vector_double>;
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
#if !defined(SMIL_WRAP_UINT32) && !defined(SMIL_WRAP_UINT)
    %template(Map_UINT) map<UINT,UINT>;
    %template(Map_UINT_double) map<UINT,double>;
    %template(Map_UINT_Vector_double) map<UINT,Vector_double>;
    %template(Map_UINT_Vector_UINT) map<UINT,Vector_UINT>;
#endif

#ifdef USE_64BIT_IDS
    %template(Map_SIZE_T) map<size_t,size_t>;
#endif // USE_64BIT_IDS
    
    %template(Map_UINT_Vector_UINT8) map< UINT, vector<UINT8> >;
    %template(Map_UINT_Vector_UINT16) map< UINT, vector<UINT16> >;
    
    TEMPLATE_WRAP_CLASS_2T_CROSS(map, Map)
    
#if !defined(SMIL_WRAP_UINT32) && !defined(SMIL_WRAP_UINT)
    TEMPLATE_WRAP_CLASS_2T_FIX_FIRST(map, UINT, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, UINT, Map)
#endif

#ifndef SMIL_WRAP_double
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, double, Map)
#endif
    
#ifndef SMIL_WRAP_RGB
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, RGB, Map)
#endif // SMIL_WRAP_RGB
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, Vector_double, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, Vector_UINT, Map)
    TEMPLATE_WRAP_CLASS_2T_FIX_SECOND(map, Box, Map)
}

#endif // SWIGXML

//////////////////////////////////////////////////////////
// Core Instance
//////////////////////////////////////////////////////////

%include "DCpuID.h"
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

// Import smilGui for viewers stuff
%import smilGui.i


%ignore smil::Image::operator[];

#ifdef SWIGPYTHON
%ignore smil::Image::getPixels;
%ignore smil::Image::getLines;
%ignore smil::Image::getSlices;
%ignore smil::Image::toArray;
%ignore smil::Image::fromArray;
%ignore smil::Image::toCharArray;
%ignore smil::Image::fromCharArray;
%ignore smil::Image::toIntArray;
%ignore smil::Image::fromIntArray;

%extend smil::Image
{
    T __getitem__(size_t i) { return self->getPixel(i); }
    RES_T __setitem__(size_t i, T val) { return self->setPixel(i, val); }
}
#endif // SWIGPYTHON

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
    TEMPLATE_WRAP_CLASS(ResImage, ResImage);
    TEMPLATE_WRAP_FUNC(createImage);
    TEMPLATE_WRAP_FUNC(castBaseImage);
    TEMPLATE_WRAP_CLASS(SharedImage, SharedImage);
    
    TEMPLATE_WRAP_SUPPL_CLASS(Image, Image);
    TEMPLATE_WRAP_SUPPL_FUNC(createImage);
    TEMPLATE_WRAP_SUPPL_FUNC(castBaseImage);
    TEMPLATE_WRAP_SUPPL_CLASS(SharedImage, SharedImage);
}


//////////////////////////////////////////////////////////
// Misc
//////////////////////////////////////////////////////////

%{
#include "DGraph.hpp"
%}

%include "DGraph.hpp"

#ifndef SWIGXML
namespace std 
{
    %template(EdgeVector_UINT) std::vector< smil::Edge<size_t> >;
}
#endif // SWIGXML

namespace smil
{
    // Base (size_t) Edge
    %template(Edge_UINT) Edge<size_t>;

    // Base (UINT) Graph
    %template(Graph_UINT) Graph<size_t,size_t>;
    // Base (UINT) MST
    %template(graphMST) graphMST<Graph<size_t,size_t> >;
}

