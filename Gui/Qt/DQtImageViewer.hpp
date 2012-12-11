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


#ifndef _D_QT_IMAGE_VIEWER_HPP
#define _D_QT_IMAGE_VIEWER_HPP

#include <QApplication>
#include <QGraphicsSceneEvent>

#include "Gui/include/private/DImageViewer.hpp"
#include "DTypes.h"

#include "Gui/Qt/ImageViewerWidget.h"
#include "Gui/Qt/ImageViewerApp.h"

#define BASE_QT_VIEWER ImageViewerWidget

template <class T> class Image;

template <class T>
class QtImageViewer : public ImageViewer<T>, public BASE_QT_VIEWER
{
public:
    typedef ImageViewer<T> parentClass;
    QtImageViewer();
    QtImageViewer(Image<T> *im);
    ~QtImageViewer();
    
    virtual void setImage(Image<T> *im);
    virtual void hide();
    virtual void show();
    virtual void showLabel();
    virtual bool isVisible();
    virtual void setName(const char *_name);
    virtual void update();
    void updateIcon()
    {
	if (!this->image)
	  return;
	
	BASE_QT_VIEWER::updateIcon();
    }
    virtual void drawOverlay(Image<T> &im);
    virtual void clearOverlay() { BASE_QT_VIEWER::clearOverlay(); }
    virtual void setCurSlice(int)
    {
        this->update();
    }
    
    virtual void setLabelImage(bool val);
    
    
protected:
    virtual void displayPixelValue(size_t x, size_t y, size_t z);
    virtual void displayMagnifyView(size_t x, size_t y, size_t z);
    virtual void drawImage();
//     ImageViewerWidget *qtViewer;
//     ImageViewer *qtViewer;
    virtual void dropEvent(QDropEvent *de);
};



#endif // _D_QT_IMAGE_VIEWER_HPP
