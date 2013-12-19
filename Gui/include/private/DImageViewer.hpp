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


#ifndef _D_IMAGE_VIEWER_HPP
#define _D_IMAGE_VIEWER_HPP

#include "Core/include/DColor.h"
#include "Gui/include/DBaseImageViewer.h"

namespace smil
{

    template <class T> class Image;
    
   /**
    * \ingroup Gui
    */
    /*@{*/

    /**
    * Base image viewer.
    * 
    */
    template <class T>
    class ImageViewer : public BaseImageViewer
    {
    public:
	typedef BaseImageViewer parentClass;
	friend class Image<T>;
	
	ImageViewer()
	  : BaseImageViewer("ImageViewer"),
	    image(NULL), 
	    labelImage(false),
	    onOverlayModified(Signal(this))
	{
	    imSize[0] = imSize[1] = imSize[2] = 0;
	}
	
	ImageViewer(Image<T> &im)
	  : BaseImageViewer("ImageViewer"),
	    image(NULL),
	    labelImage(false),
	    onOverlayModified(Signal(this))
	{
	    imSize[0] = imSize[1] = imSize[2] = 0;
	    setImage(im);
	}
	
	virtual void setImage(Image<T> &im)
	{
	    if (image)
	      disconnect();
	    
	    image = &im;
	    image->getSize(imSize);
	    this->setName(image->getName());
	    
	    if (&im)
	      image->onModified.connect(&this->updateSlot);
	}
	virtual Image<T> *getImage()
	{
	    return this->image;
	}
	virtual void disconnect()
	{
	    if (image)
	      image->onModified.disconnect(&this->updateSlot);
	    image = NULL;
	}
	
	virtual void show() {}
	virtual void showLabel() {}
	virtual void hide() {}
	virtual bool isVisible() { return false; }
	virtual void setName(const char *_name) { parentClass::setName(_name); }
	virtual void update()
	{
	    if (!this->image)
	      return;
	    
	    size_t newSize[3];
	    this->image->getSize(newSize);
	    if (imSize[0]!=newSize[0] || imSize[1]!=newSize[1] || imSize[2]!=newSize[2])
	    {
		this->setImage(*this->image);
	    }
	    this->setName(image->getName());
	    
	    if (this->isVisible())
	      this->drawImage();
	}
	virtual void drawOverlay(Image<T> &) {}
	virtual void clearOverlay() {}
	virtual RES_T getOverlay(Image<T> &img) { return RES_ERR; }
	
	Signal onOverlayModified;
	
	//! Set the color table as a 8bits RGB map (keys between 0 and 255)
	virtual void setLookup(const map<UINT8,RGB> &lut) {}
	virtual void resetLookup() {}
	
    protected:
	virtual void drawImage() {}
	virtual void onSizeChanged(size_t width, size_t height, size_t depth) {}
	Image<T> *image;
	bool labelImage;
	
    private:
	size_t imSize[3];
    };

    /*@}*/
    
} // namespace smil


#endif // _D_BASE_IMAGE_VIEWER_H
