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


#ifndef _DRGB_H
#define _DRGB_H

#include "DColor.h"
#include "DImage_RGB.h"
#include "DLineArith_RGB.h"

namespace smil
{

    template <>
    void QtImageViewer<RGB>::setImage(Image<RGB> &im);
    template <>
    void QtImageViewer<RGB>::drawImage();
    template <>
    void QtImageViewer<RGB>::displayPixelValue(size_t x, size_t y, size_t z);

    template <>
    void QtImageViewer<RGB>::drawOverlay(Image<RGB> &im);
    
    template <>
    RES_T readVTK<RGB>(const char *filename, Image<RGB> &image);

    template <>
    RES_T writeVTK<RGB>(const Image<RGB> &image, const char *filename, bool binary);

    
    

#if defined SWIGPYTHON and defined USE_NUMPY
    template <>
    PyObject * Image<RGB>::getNumArray(bool c_contigous)
    {
    }
#endif // defined SWIGPYTHON and defined USE_NUMPY

} // namespace smil

#endif // _DRGB_H

