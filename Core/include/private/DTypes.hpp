/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_TYPES_HPP
#define _D_TYPES_HPP

#ifndef _MSC_VER
#include <stdint.h>
#endif // _MSC_VER
#include <limits>

#include "Core/include/private/DMemory.hpp"

#include <sstream>

using namespace std;

#define NEWTDEFS   0

namespace smil
{
    typedef int            INT;
    typedef long           LONG;
    typedef unsigned int   UINT;
    typedef unsigned long  ULONG;
#if NEWTDEFS
    typedef uint8_t        UINT8;
    typedef uint16_t       UINT16;
    typedef uint32_t       UINT32;
#else
    typedef unsigned char  UINT8;
    typedef unsigned short UINT16;
    typedef unsigned int   UINT32;
#endif
#ifdef _MSC_VER
        typedef unsigned __int64 UINT64;
#else
    typedef uint64_t       UINT64;
#endif

    // typedef unsigned char __attribute__ ((vector_size (16))) alUINT8;

	// Why this directive for msvc (windows) ?
    //#ifndef _MSC_VER
    typedef signed char INT8;
    //#endif // _MSC_VER
    typedef short INT16;
    typedef int INT32;


    #ifndef CHAR_BIT
    #define CHAR_BIT 8
    #endif

    enum DType
    {
        DtUINT8, DtUINT16, DtUINT32, DtUINT64, DtINT, DtUINT
    };

    
    template <class T>
    struct ImDtTypes
    {
        typedef T pixelType;
        typedef pixelType *lineType;
#ifndef SWIG        
        typedef pixelType * __restrict restrictLineType;
#endif // SWIG        
        typedef lineType *sliceType;
        typedef sliceType *volType;
        
        typedef std::vector<T, Allocator<T> > vectorType;
//         typedef std::vector<T> vectorType;
        typedef std::vector< vectorType > matrixType;
        
        typedef double floatType;
        
        static inline pixelType min() { return numeric_limits<T>::min(); }
        static inline pixelType max() { return numeric_limits<T>::max(); }
        static inline size_t cardinal() { return max()-min()+1; }
        static inline lineType createLine(size_t lineLen) { return createAlignedBuffer<T>(lineLen); }
        static inline void deleteLine(lineType line) { deleteAlignedBuffer<T>(line); }
        static inline size_t ptrOffset(lineType p, size_t n=SIMD_VEC_SIZE) { return ((size_t)p) & (n-1); }
        static inline std::string toString(const T &val)
        {
            stringstream str;
            str << double(val);
            return str.str();
        }
    };



    template <class T>
    inline const char *getDataTypeAsString(T * /*val*/ =(T*)NULL)
    {
        return "Unknown";
    }


    #define DECL_DATA_TYPE_STR(_type) \
    template <> \
    inline const char *getDataTypeAsString(_type *) { return #_type; }

    DECL_DATA_TYPE_STR(UINT8)
    DECL_DATA_TYPE_STR(UINT16)
    DECL_DATA_TYPE_STR(INT8)
    DECL_DATA_TYPE_STR(INT16)
    DECL_DATA_TYPE_STR(int)
    DECL_DATA_TYPE_STR(float)
    DECL_DATA_TYPE_STR(double)

} // namespace smil


#endif // _D_TYPES_HPP
