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
#include <QList>
#include <QUrl>

#ifdef USE_QWT
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_series_data.h>
#endif // USE_QWT


#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DImageHistogram.hpp"

namespace smil
{
    template <class T>
    QtImageViewer<T>::QtImageViewer()
      : BASE_QT_VIEWER(NULL),
	histoPlot(NULL)
    {
    }

    template <class T>
    QtImageViewer<T>::QtImageViewer(Image<T> *im)
      : ImageViewer<T>(im), BASE_QT_VIEWER(NULL),
	histoPlot(NULL)
    {
	setImage(im);
    }

    template <class T>
    QtImageViewer<T>::~QtImageViewer()
    {
	close();

    #ifdef USE_QWT
	if (histoPlot)
	  delete histoPlot;
    #endif // USE_QWT
    }


    template <class T>
    void QtImageViewer<T>::setImage(Image<T> *im)
    {
	ImageViewer<T>::setImage(im);
	BASE_QT_VIEWER::setImageSize(im->getWidth(), im->getHeight(), im->getDepth());
	if (im->getName()!=string(""))
	  setName(this->image->getName());
    }

    template <class T>
    void QtImageViewer<T>::show()
    {
	BASE_QT_VIEWER::showNormal();
	BASE_QT_VIEWER::raise();
	BASE_QT_VIEWER::activateWindow();
    }

    template <class T>
    void QtImageViewer<T>::showLabel()
    {
	this->setLabelImage(true);
	BASE_QT_VIEWER::showNormal();
	BASE_QT_VIEWER::raise();
	BASE_QT_VIEWER::activateWindow();
	update();
    }

    template <class T>
    void QtImageViewer<T>::hide()
    {
	BASE_QT_VIEWER::hide();
    }

    template <class T>
    bool QtImageViewer<T>::isVisible()
    {
	return BASE_QT_VIEWER::isVisible();
    }

    template <class T>
    void QtImageViewer<T>::setName(const char *_name)
    {
	parentClass::setName(_name);
	QString buf = _name + QString(" (") + QString(parentClass::image->getTypeAsString()) + QString(")");
	BASE_QT_VIEWER::setName(buf);
    }

    template <class T>
    void QtImageViewer<T>::setLabelImage(bool val)
    {
	if (parentClass::labelImage==val)
	  return;
	
	BASE_QT_VIEWER::setLabelImage(val);
	parentClass::labelImage = val;
	    
	update();
    }

    template <class T>
    void QtImageViewer<T>::update()
    {
	if (!this->image)
	  return;
	
	drawImage();
	BASE_QT_VIEWER::dataChanged();

	BASE_QT_VIEWER::update();

    #ifdef USE_QWT    
	if (histoPlot && histoPlot->isVisible())
	  displayHistogram(true);
    #endif // USEçQWT    
    //     qApp->processEvents();
    }


    template <class T>
    void QtImageViewer<T>::drawImage()
    {
	typename Image<T>::sliceType lines = this->image->getSlices()[slider->value()];
	
	size_t w = this->image->getWidth();
	size_t h = this->image->getHeight();
	
	UINT8 *destLine;
	double coeff;
	
	if (parentClass::labelImage)
	  coeff = 1.0;
	else
	  coeff = double(numeric_limits<UINT8>::max()) / ( double(numeric_limits<T>::max()) - double(numeric_limits<T>::min()) );

	for (size_t j=0;j<h;j++,lines++)
	{
	    typename Image<T>::lineType pixels = *lines;
	    
	    destLine = this->qImage->scanLine(j);
	    for (size_t i=0;i<w;i++)
    // 	  pixels[i] = 0;
		destLine[i] = (UINT8)(coeff * (double(pixels[i]) - double(numeric_limits<T>::min())));
	}
    }


    // Specialization for UINT8 type (faster)
    template <>
    void QtImageViewer<UINT8>::drawImage();


    #ifdef SMIL_WRAP_Bit
    // template <>
    // void QtImageViewer<Bit>::drawImage();
    #endif // SMIL_WRAP_Bit


    template <class T>
    void QtImageViewer<T>::drawOverlay(Image<T> &im)
    {
	size_t w = im.getWidth();
	size_t h = im.getHeight();
	
	typename Image<T>::sliceType lines = im.getSlices()[slider->value()];
	typename Image<T>::lineType pixels;
	
	if (qOverlayImage)
	  delete qOverlayImage;
	
	qOverlayImage = new QImage(w, h, QImage::Format_ARGB32_Premultiplied);
	qOverlayImage->setColorTable(overlayColorTable);
	qOverlayImage->fill(Qt::transparent);

	QRgb *destLine;
	  
	for (size_t j=0;j<im.getHeight();j++)
	{
	    destLine = (QRgb*)(this->qOverlayImage->scanLine(j));
	    pixels = *lines++;
	    for (size_t i=0;i<im.getWidth();i++)
	    {
	      if (pixels[i]!=0)
		destLine[i] = overlayColorTable[(UINT8)pixels[i]];
	    }
	}
	    
	BASE_QT_VIEWER::overlayDataChanged();
	BASE_QT_VIEWER::update();
    }

    template <class T>
    void QtImageViewer<T>::setLookup(const map<UINT8,RGB> &lut)
    {
	baseColorTable.clear();
	map<UINT8,RGB>::const_iterator it;
	for (int i=0;i<256;i++)
	{
	  it = lut.find(UINT8(i));
	  if (it!=lut.end())
	    baseColorTable.append(qRgb((*it).second.r, (*it).second.g, (*it).second.b));
	  else
	    baseColorTable.append(qRgb(0, 0, 0));
	}
	qImage->setColorTable(baseColorTable);
	showNormal();
	update();
    }
    
    template <class T>
    void QtImageViewer<T>::resetLookup()
    {
	baseColorTable.clear();
	for (int i=0;i<256;i++)
	  baseColorTable.append(qRgb(i, i, i));
	qImage->setColorTable(baseColorTable);
	showNormal();
	update();
    }
	
    template <class T>
    void QtImageViewer<T>::displayPixelValue(size_t x, size_t y, size_t z)
    {
	T pixVal;

	pixVal = this->image->getPixel(x, y, z);
	QString txt = "(" + QString::number(x) + ", " + QString::number(y);
	if (this->image->getDepth()>1)
	  txt = txt + ", " + QString::number(z);
	txt = txt + ") " + QString::number(pixVal);
	valueLabel->setText(txt);
	valueLabel->adjustSize();
    }


    template <class T>
    void QtImageViewer<T>::displayMagnifyView(size_t x, size_t y, size_t z)
    {
	magnView->displayAt(x, y);
    //   return;

	int gridSize = magnView->getGridSize();
	int halfGrid = (gridSize-1)/2;
	
	int yi = y-halfGrid;

	int imW = qImage->width();
	int imH = qImage->height();

	typename ImDtTypes<T>::sliceType pSlice = parentClass::image->getSlices()[z];
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

    template <class T>
    void QtImageViewer<T>::dropEvent(QDropEvent *de)
    {
	QList<QUrl> urls = de->mimeData()->urls();
	int objNbr = urls.size();

	if (objNbr==1)
	  read(urls[0].path().toStdString().c_str(), *this->image);
	else
	{
	    vector<string> files;
	    for (QList<QUrl>::iterator it=urls.begin();it!=urls.end();it++)
	      files.push_back((*it).path().toStdString());
	    read(files, *this->image);
	}
    }

#ifdef USE_QWT
    template <class T>
    void QtImageViewer<T>::displayHistogram(bool update)
    {
    if (!update && histoPlot && histoPlot->isVisible())
	{
	  histoPlot->raise();
	  histoPlot->activateWindow();
	  return;
	}
	
	if (!histoPlot)
	{
	    histoPlot = new QwtPlot();
	    histoPlot->setWindowTitle(QString(this->image->getName()) + " histogram");
	    histoPlot->setFixedSize(480,280);
	    histoPlot->setCanvasBackground(Qt::white);
	    histoPlot->setAxisScale(QwtPlot::xBottom, ImDtTypes<T>::min(), ImDtTypes<T>::max());
	}
	histoPlot->detachItems();
	
	QwtPlotCurve *curve = new QwtPlotCurve("Image Histogram");
	curve->setStyle( QwtPlotCurve::Steps );
	curve->setBrush( Qt::blue );
      
	QwtPointSeriesData *myData = new QwtPointSeriesData();
	map<T, UINT> hist = histogram(*(this->image));
      
	QVector<QPointF> samples;
	for(typename map<T,UINT>::iterator it=hist.begin();it!=hist.end();it++)
	  samples.push_back(QPointF((*it).first, (*it).second));

	myData->setSamples(samples);
	curve->setData(myData);
      
	curve->attach(histoPlot);
	
	histoPlot->replot();
	histoPlot->show();
    }
#endif // USE_QWT

} // namespace smil

#endif // _D_QT_IMAGE_VIEWER_HXX
