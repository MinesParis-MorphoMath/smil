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


#ifndef _D_QT_IMAGE_VIEWER_HPP
#define _D_QT_IMAGE_VIEWER_HPP

#include <QApplication>
#include <QGraphicsSceneEvent>

#include "DImageViewer.hpp"
#include "DTypes.h"

#include "Qt/ImageViewerWidget.h"
#include "Qt/ImageViewer.h"

#define BASE_QT_VIEWER ImageViewerWidget

template <class T> class Image;

template <class T>
class qtImageViewer : public imageViewer<T>, protected BASE_QT_VIEWER
{
public:
    typedef imageViewer<T> parentClass;
    qtImageViewer(Image<T> *im);
    ~qtImageViewer();
    virtual void hide();
    virtual void show();
    virtual void showLabel();
    virtual bool isVisible();
    virtual void setName(const char* _name);
    virtual void update();
    virtual void drawOverlay(Image<T> &im);
    virtual void clearOverlay();
    
    virtual void setLabelImage(bool val);
    
    QApplication *_qapp;
    
    virtual void imageMouseMoveEvent ( QGraphicsSceneMouseEvent * event ) 
    {
	int x = int(event->scenePos().rx());
	int y = int(event->scenePos().ry());
	T pixVal;

	pixVal = this->image->getPixel(x, y);
	valueLabel->setText("(" + QString::number(x) + ", " + QString::number(y) + ") " + QString::number(pixVal));
	valueLabel->adjustSize();
    }
protected:
    virtual void drawImage();
//     ImageViewerWidget *qtViewer;
//     ImageViewer *qtViewer;
};


template <class T>
qtImageViewer<T>::qtImageViewer(Image<T> *im)
  : imageViewer<T>(im), BASE_QT_VIEWER(NULL)
{
    setImageSize(im->getWidth(), im->getHeight());
    if (this->image->getName())
      setName(this->image->getName());
}

template <class T>
qtImageViewer<T>::~qtImageViewer()
{
    hide();
//     delete qtViewer;
}

template <class T>
void qtImageViewer<T>::show()
{
    BASE_QT_VIEWER::show();
}

template <class T>
void qtImageViewer<T>::showLabel()
{
    setLabelImage(true);
    BASE_QT_VIEWER::show();
}

template <class T>
void qtImageViewer<T>::hide()
{
     BASE_QT_VIEWER::hide();
}

template <class T>
bool qtImageViewer<T>::isVisible()
{
     return BASE_QT_VIEWER::isVisible();
}

template <class T>
void qtImageViewer<T>::setName(const char* _name)
{
    parentClass::setName(_name);
    BASE_QT_VIEWER::setName(_name);
}

template <class T>
void qtImageViewer<T>::setLabelImage(bool val)
{
    if (parentClass::labelImage==val)
      return;
    
    BASE_QT_VIEWER::setLabelImage(val);
    parentClass::labelImage = BASE_QT_VIEWER::drawLabelized;
    drawImage();
}

template <class T>
void qtImageViewer<T>::update()
{
    drawImage();
    BASE_QT_VIEWER::update();
}

template <class T>
void qtImageViewer<T>::drawImage()
{
    typename Image<T>::lineType pixels = this->image->getPixels();
    UINT w = this->image->getWidth();
    UINT h = this->image->getHeight();
    
    this->setImageSize(w, h);
    
    UINT8 *destLine;
    double coeff;
    
    if (parentClass::labelImage)
      coeff = 1.0;
    else
      coeff = double(numeric_limits<UINT8>::max()) / double(numeric_limits<T>::max());

    for (int j=0;j<h;j++)
    {
	destLine = this->qImage->scanLine(j);
	for (int i=0;i<w;i++)
	    destLine[i] = (UINT8)(coeff * double(pixels[i]));
	
	pixels += w;
    }

    if (parentClass::labelImage)
      qImage->setColorTable(labelColorTable);
    else qImage->setColorTable(baseColorTable);

    qOverlayImage->fill(Qt::transparent);
}


// Specialization for UINT8 type
template <>
void qtImageViewer<UINT8>::drawImage();


template <class T>
void qtImageViewer<T>::clearOverlay()
{
    qOverlayImage->fill(Qt::transparent);
    overlayPixmap->setPixmap(QPixmap::fromImage(*qOverlayImage));

    BASE_QT_VIEWER::update();
}

template <class T>
void qtImageViewer<T>::drawOverlay(Image<T> &im)
{
    qOverlayImage->fill(Qt::transparent);

    typename Image<T>::lineType pixels = *im.getSlices()[0];
    UINT pixNbr = im.getWidth()*im.getHeight();
      
    for (int j=0;j<im.getHeight();j++)
      for (int i=0;i<im.getWidth();i++)
      {
	if (*pixels!=0)
	  qOverlayImage->setPixel(i, j, overlayColorTable[(UINT8)*pixels]);
	pixels++;
      }
	
    overlayPixmap->setPixmap(QPixmap::fromImage(*qOverlayImage));

    BASE_QT_VIEWER::update();
}


#ifdef SMIL_WRAP_Bit
template <>
void qtImageViewer<Bit>::loadFromData(ImDtTypes<Bit>::lineType pixels, UINT w, UINT h);
#endif // SMIL_WRAP_Bit


#endif // _D_QT_IMAGE_VIEWER_HPP
