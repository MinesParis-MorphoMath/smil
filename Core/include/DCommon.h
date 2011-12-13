/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
 *     * Neither the name of the University of California, Berkeley nor the
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


#ifndef _DCOMMON_H
#define _DCOMMON_H


#include <string.h>
#include <memory>
#include <limits>
#include <vector>
#include <stdarg.h>


#include "DTypes.hpp"
// #include "auto_ptr.h"
// #include <boost/smart_ptr.hpp>

using namespace std;


#define VERBOSE 1

#if VERBOSE > 1
#define MESSAGE(msg) cout << msg <<  endl;
#else // VERBOSE 
#define MESSAGE(msg)
#endif // VERBOSE 



#define INLINE inline

#ifdef _MSC_VER
#ifdef smilCore_EXPORTS
// the dll exports
#define _SMIL __declspec(dllexport)
#else // smilCore_EXPORTS
// the exe imports
#define _SMIL __declspec(dllimport)
#endif // smilCore_EXPORTS
#else // _MSC_VER
#define _SMIL
#endif // _MSC_VER



#define SMART_POINTER(T) boost::shared_ptr< T >
#define SMART_IMAGE(T) SMART_POINTER( D_Image< T > )

#define D_DEFAULT_IMAGE_WIDTH 512
#define D_DEFAULT_IMAGE_HEIGHT 512
#define D_DEFAULT_IMAGE_DEPTH 1

#define D_DEFAULT_OUT_PIXEL_VAL 0




#endif // _DCOMMON_H

