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

#ifdef BUILD_GUI


#include "DQtImageViewer.hpp"
#include "Qt/ImageViewerWidget.h"
#include "DImage.hpp"


template <>
void qtImageViewer<UINT8>::loadFromData(Image<UINT8>::lineType pixels, UINT w, UINT h)
{
    qtViewer->loadFromData(pixels, w, h);
}

template <>
void qtImageViewer<UINT16>::loadFromData(Image<UINT16>::lineType pixels, UINT w, UINT h)
{
    qtViewer->setImageSize(w, h);
    
    UINT8 *destLine;
    double coeff = double(numeric_limits<UINT8>::max()) / double(numeric_limits<UINT16>::max());

    for (int j=0;j<h;j++)
    {
	destLine = qtViewer->image->scanLine(j);
	for (int i=0;i<w;i++)
	    destLine[i] = (UINT8)(coeff * double(pixels[i]));
	
	pixels += w;
    }

    qtViewer->dataChanged();
}


#ifdef SMIL_WRAP_BIN

// #include "DBitArray.h"

template <>
void qtImageViewer<BIN>::loadFromData(Image<BIN>::lineType pixels, UINT w, UINT h)
{
    qtViewer->setImageSize(w, h);
    
    const BIN *lIn;
    UINT8 *lOut, *lEnd;
    UINT bCount = (w-1)/BIN::SIZE + 1; 

    for (int j=0;j<h;j++)
    {
	lIn = pixels + j*bCount;
	lOut = qtViewer->image->scanLine(j);
	lEnd = lOut + w;
	
	for (int b=0;b<bCount;b++,lIn++)
	{
	  BIN_TYPE bVal = (*lIn).val;
	  
	  for (int i=0;i<BIN::SIZE;i++,lOut++)
	  {
	    if (lOut==lEnd)
	      break;
	    *lOut = bVal & (1 << i) ? 255 : 0;
	  }
	}
    }

    qtViewer->dataChanged();
}

#endif // SMIL_WRAP_BIN


#ifdef SMIL_WRAP_Bit

// #include "DBitArray.h"

template <>
void qtImageViewer<Bit>::loadFromData(Image<Bit>::lineType pixels, UINT w, UINT h)
{
    qtViewer->setImageSize(w, h);
    
    BIN* data = (BIN*)pixels.intArray;
    const BIN *lIn;
    UINT8 *lOut, *lEnd;
    UINT bCount = (w-1)/BIN::SIZE + 1; 

    for (int j=0;j<h;j++)
    {
	lIn = data + j*bCount;
	lOut = qtViewer->image->scanLine(j);
	lEnd = lOut + w;
	
	for (int b=0;b<bCount;b++,lIn++)
	{
	  BIN_TYPE bVal = (*lIn).val;
	  
	  for (int i=0;i<BIN::SIZE;i++,lOut++)
	  {
	    if (lOut==lEnd)
	      break;
	    *lOut = bVal & (1 << i) ? 255 : 0;
	  }
	}
    }

    qtViewer->dataChanged();
}

#endif // SMIL_WRAP_Bit


#endif // BUILD_GUI
