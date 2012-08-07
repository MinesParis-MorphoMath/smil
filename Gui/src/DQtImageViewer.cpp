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

#ifdef USE_QT


#include "DQtImageViewer.hpp"
#include "Qt/ImageViewerWidget.h"
#include "DImage.hpp"


template <>
void qtImageViewer<UINT8>::drawImage()
{
    Image<UINT8>::sliceType lines = this->image->getSlices()[slider->value()];
    
    UINT w = this->image->getWidth();
    UINT h = this->image->getHeight();
    UINT d = this->image->getDepth();
    
    this->setImageSize(w, h, d);

    for (int j=0;j<h;j++, lines++)
        memcpy(qImage->scanLine(j), *lines, sizeof(uchar) * w);
}



#ifdef SMIL_WRAP_BIN

// #include "DBitArray.h"

template <>
void qtImageViewer<BIN>::drawImage()
{
    Image<BIN>::lineType pixels = this->image->getPixels();
    UINT w = this->image->getWidth();
    UINT h = this->image->getHeight();
    
    this->setImageSize(w, h);
    
    const BIN *lIn;
    UINT8 *lOut, *lEnd;
    UINT bCount = (w-1)/BIN::SIZE + 1; 

    for (int j=0;j<h;j++)
    {
	lIn = pixels + j*bCount;
	lOut = this->qImage->scanLine(j);
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
}

#endif // SMIL_WRAP_BIN


#ifdef SMIL_WRAP_Bit

// #include "DBitArray.h"

template <>
void qtImageViewer<Bit>::drawImage()
{
    Image<Bit>::lineType pixels = this->image->getPixels();
    UINT w = this->image->getWidth();
    UINT h = this->image->getHeight();
    
    this->setImageSize(w, h);
    
    BIN* data = (BIN*)pixels.intArray;
    const BIN *lIn;
    UINT8 *lOut, *lEnd;
    UINT bCount = (w-1)/BIN::SIZE + 1; 

    for (int j=0;j<h;j++)
    {
	lIn = data + j*bCount;
	lOut = this->qImage->scanLine(j);
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
}

#endif // SMIL_WRAP_Bit


#endif // USE_QT
