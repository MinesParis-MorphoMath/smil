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

#include "Gui/include/DBaseImageViewer.h"

template <class T> class Image;

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
      : image(NULL), labelImage(false)
    {
    }
    
    ImageViewer(Image<T> *im)
      : labelImage(false)
    {
	setImage(im);
    }
    
    virtual void setImage(Image<T> *im)
    {
	image = im;
	
	if (!im)
	  return;
	
	setName(im->getName());
    }
    void connect(Image<T> *im)
    {
	if (image)
	  disconnect(image);
	this->setImage(im);
	image->onModified.connect(&this->updateSlot);
	update();
    }
    virtual void disconnect(Image<T> *im)
    {
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
	if (image)
	  drawImage();
    }
    virtual void drawOverlay(Image<T> &im) {}
    virtual void clearOverlay() {}
    
protected:
    Image<T> *getImage() { return image; }
    virtual void drawImage() {}
    bool labelImage;
    Image<T> *image;
};



#endif // _D_BASE_IMAGE_VIEWER_H
