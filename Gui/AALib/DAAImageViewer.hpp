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


#ifndef _D_AA_IMAGE_VIEWER_HPP
#define _D_AA_IMAGE_VIEWER_HPP

#include "DImageViewer.hpp"
#include "DImageTransform.hpp"
#include "DTypes.h"

#include <aalib.h>
template <class T> class Image;

template <class T>
class aaImageViewer : public imageViewer<T>
{
public:
    typedef imageViewer<T> parentClass;
    aaImageViewer();
    aaImageViewer(Image<T> *im);
    ~aaImageViewer();
    virtual void hide();
    virtual void show();
    virtual bool isVisible();
    virtual void setName(const char* _name);
    virtual void clearOverlay() { }

protected:
    aa_context *context;
    int createContext();
    virtual void drawImage();
};


template <class T>
aaImageViewer<T>::aaImageViewer()
{
    context = NULL;
}

template <class T>
aaImageViewer<T>::aaImageViewer(Image<T> *im)
  : imageViewer<T>(im)
{
    context = NULL;
}

template <class T>
int aaImageViewer<T>::createContext()
{
    context = aa_autoinit(&aa_defparams);
    if(context == NULL) 
    {
      fprintf(stderr,"Cannot initialize AA-lib. Sorry.\n");
      return -1;
    }
    return 0;
}

template <class T>
aaImageViewer<T>::~aaImageViewer()
{
    hide();
}


template <class T>
void aaImageViewer<T>::show()
{
    drawImage();
}

template <class T>
void aaImageViewer<T>::hide()
{
    if (context)      
      aa_close(context);
    context = NULL;
}

template <class T>
void aaImageViewer<T>::drawImage()
{
    if (!context)
      createContext();
    
    aa_resize(context);
    
    double imW = this->image->getWidth();
    double imH = this->image->getHeight();
    double imR = imW / imH;
    
    double scrW = aa_imgwidth(context);
    double scrH = aa_imgheight(context);
    double scrR = scrW / scrH;
    
    UINT w, h;
    // find dimensions to fit screen
    if (scrR > imR)
    {
	w = imW * scrH / imH;
	h = scrH;
    }
    else
    {
	w = scrW;
	h = imH * scrW / imW;
    }
    w *= aa_imgheight(context) / aa_scrheight(context);
    
    Image<T> tmpIm(w, h);
    resize(*this->image, tmpIm);
    
    unsigned char *data = aa_image(context);
    typename Image<T>::lineType pixels = tmpIm.getPixels();
    double coeff = double(numeric_limits<UINT8>::max()) / double(numeric_limits<T>::max());
    
    for (int j=0;j<scrH;j++)
      for (int i=0;i<scrW;i++,data++)
      {
	if (i<w && j<h)
	  *data = (UINT8)(coeff * double(*pixels++));
	else *data = 0;
      }
      
    aa_render(context, &aa_defrenderparams, 0, 0, aa_scrwidth (context), aa_scrheight (context));
    aa_flush(context);
}

template <class T>
bool aaImageViewer<T>::isVisible()
{
    return context!=NULL;
}

template <class T>
void aaImageViewer<T>::setName(const char* _name)
{
}




#endif // _D_AA_IMAGE_VIEWER_HPP
