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


#ifndef _D_IMAGE_IO_RAW_HPP
#define _D_IMAGE_IO_RAW_HPP


#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "DTypes.hpp"
#include "DImage.h"

using namespace std;

template <class T>
RES_T readRAW(const char *filename, UINT width, UINT height, UINT depth, Image<T> &image)
{
    FILE *fp = NULL;

    /* open image file */
    fp = fopen (filename, "rb");
    if (!fp)
    {
        fprintf (stderr, "error: couldn't open \"%s\"!\n", filename);
        return RES_ERR;
    }


    image.setSize(width, height, depth);
//   image->allocate();

    size_t result = fread(image.getVoidPointer(), image.getAllocatedSize(), 1, fp);

    fclose (fp);
    
    image.modified();
    
    if (result==image.getAllocatedSize())
      return RES_OK;
    else return RES_ERR;
}

template <class T>
RES_T writeRAW(Image<T> &image, const char *filename)
{
    FILE *fp = NULL;

    /* open image file */
    fp = fopen (filename, "wb");
    if (!fp)
    {
        fprintf (stderr, "error: couldn't open \"%s\"!\n", filename);
        return RES_ERR;
    }

    size_t result = fwrite(image.getVoidPointer(), image.getAllocatedSize(), 1, fp);

    fclose (fp);
    
    if (result==image.getAllocatedSize())
      return RES_OK;
    else return RES_ERR;
}



#endif // _D_IMAGE_IO_RAW_H
