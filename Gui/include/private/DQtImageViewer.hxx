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


#ifndef _D_QT_IMAGE_VIEWER_HXX
#define _D_QT_IMAGE_VIEWER_HXX

#include <QApplication>
#include <QGraphicsSceneEvent>

#include "DImage.hpp"


template <class T>
qtImageViewer<T>::qtImageViewer(Image<T> *im)
  : imageViewer<T>(im), BASE_QT_VIEWER(NULL)
{
    setImageSize(im->getWidth(), im->getHeight());
    if (this->image->getName()!="")
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
    update();
}

template <class T>
void qtImageViewer<T>::showLabel()
{
    this->setLabelImage(true);
    BASE_QT_VIEWER::show();
    update();
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
void qtImageViewer<T>::setName(string _name)
{
    parentClass::setName(_name);
    QString buf = _name.c_str() + QString(" (") + QString(parentClass::image->getTypeAsString()) + QString(")");
    BASE_QT_VIEWER::setName(buf);
}

template <class T>
void qtImageViewer<T>::setLabelImage(bool val)
{
    if (parentClass::labelImage==val)
      return;
    
    BASE_QT_VIEWER::setLabelImage(val);
    parentClass::labelImage = val;
    
    if (val)
      qImage->setColorTable(labelColorTable);
    else qImage->setColorTable(baseColorTable);
    
    parentClass::dataModified = true;
    BASE_QT_VIEWER::dataChanged();
}

template <class T>
void qtImageViewer<T>::update()
{
    if (parentClass::dataModified)
    {
	drawImage();
	BASE_QT_VIEWER::dataChanged();
	parentClass::dataModified = false;
    }
    BASE_QT_VIEWER::update();
    qApp->processEvents();
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
}


// Specialization for UINT8 type (faster)
template <>
void qtImageViewer<UINT8>::drawImage();


#ifdef SMIL_WRAP_Bit
template <>
void qtImageViewer<Bit>::drawImage();
#endif // SMIL_WRAP_Bit


template <class T>
void qtImageViewer<T>::drawOverlay(Image<T> &im)
{
    UINT w = im.getWidth();
    UINT h = im.getHeight();
    
    if (qOverlayImage)
      delete qOverlayImage;
    
    qOverlayImage = new QImage(w, h, QImage::Format_ARGB32_Premultiplied);
//     qOverlayImage.setColorTable(overlayColorTable);
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
	
    BASE_QT_VIEWER::overlayDataChanged();
    BASE_QT_VIEWER::update();
}

template <class T>
void qtImageViewer<T>::displayPixelValue(UINT x, UINT y)
{
    T pixVal;

    pixVal = this->image->getPixel(x, y);
    valueLabel->setText("(" + QString::number(x) + ", " + QString::number(y) + ") " + QString::number(pixVal));
    valueLabel->adjustSize();
}


template <class T>
void qtImageViewer<T>::displayMagnifyView(UINT x, UINT y)
{
    magnView->displayAt(x, y);
//   return;

    int gridSize = magnView->getGridSize();
    int halfGrid = (gridSize-1)/2;
    
    int xi = x-halfGrid;
    int yi = y-halfGrid;

    int imW = qImage->width();
    int imH = qImage->height();

    double s = magnView->getScaleFactor() / gridSize;

    typename ImDtTypes<T>::sliceType pSlice = parentClass::image->getSlices()[0];
    typename ImDtTypes<T>::lineType pLine;
    T pVal;
    
    QGraphicsTextItem *textItem;
    QList<QGraphicsTextItem*>::Iterator txtIt = magnView->getTextItemList()->begin();
    
    QColor lightCol = QColor::fromRgb(255,255,255);
    QColor darkCol = QColor::fromRgb(0,0,0);
    T lightThresh = double(ImDtTypes<T>::max()-ImDtTypes<T>::min()) * 0.55;

    for (int j=0;j<gridSize;j++,yi++)
    {
        if (yi>=0 && yi<imH)
            pLine = pSlice[yi];
        else pLine = NULL;

        for (int i=0,xi=x-gridSize/2; i<gridSize; i++,xi++)
        {
            textItem = *txtIt++;
            if (pLine && xi>=0 && xi<imW)
            {
                pVal = pLine[xi];
                if (pVal<lightThresh)
                    textItem->setDefaultTextColor(lightCol);
                else
                    textItem->setDefaultTextColor(darkCol);
                textItem->setPlainText(QString::number(pVal));
            }
            else textItem->setPlainText("");

        }
    }
}


#endif // _D_QT_IMAGE_VIEWER_HXX
