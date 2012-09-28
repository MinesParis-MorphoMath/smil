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


#ifndef _D_MEASURES_HPP
#define _D_MEASURES_HPP

/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

#include "DImage.hpp"

template <class T>
struct measOp
{
    virtual void _exec(typename Image<T>::lineType pixels, UINT y, UINT size) = 0;
};

template <class T>
struct baryOp : public measOp<T>
{
    baryOp()
      : xSum(0), ySum(0), tSum(0)
    {
    }
    virtual void _exec(typename Image<T>::lineType pixels, UINT y, UINT size)
    {
        for (UINT i=0;i<size;i++)
	  if (pixels[i]!=0)
	  {
	      xSum = pixels[i] * i;
	      ySum = pixels[i] * y;
	      tSum++;		  
	  }
    }
    double xSum;
    double ySum;
    double tSum;
};

template <class T>
RES_T measBarycenter(Image<T> &im, UINT *xc, UINT *yc)
{
    typename Image<T>::volType slices = im.getSlices();
    typename Image<T>::sliceType lines;
    
    measOp<T> *op = new baryOp<T>();
    
    UINT imSize[3];
    im.getSize(imSize);
    
    for (UINT z=0;z<imSize[2];z++)
    {
	lines = *slices++;
#pragma omp parallel for
	for (UINT y=0;y<imSize[1];y++)
	  op->_exec(*lines++, y, imSize[0]);
    }
    
//     *xc = op.xSum / op.tSum;
//     *yc = op.ySum / op.tSum;
    
    return RES_OK;
}



/** @}*/

#endif // _D_MEASURES_HPP

