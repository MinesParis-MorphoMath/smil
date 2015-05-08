/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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
#include <QMimeData>

#ifdef USE_QWT
#include "PureQt/PlotWidget.h"
#endif // USE_QWT

#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DImageHistogram.hpp"
#include "Base/include/DImageDraw.h"
#include "IO/include/private/DImageIO.hpp"
#include "Core/include/DColor.h"

namespace smil
{
    template <class T>
    QtImageViewer<T>::QtImageViewer()
      : BASE_QT_VIEWER(NULL)
    {
    #ifdef USE_QWT
        histoPlot = NULL;
        profilePlot = NULL;
    #endif // USE_QWT
    }

    template <class T>
    QtImageViewer<T>::QtImageViewer(Image<T> &im)
      : ImageViewer<T>(im),
        BASE_QT_VIEWER(NULL)
    {
    #ifdef USE_QWT
        histoPlot = NULL;
        profilePlot = NULL;
    #endif // USE_QWT
        setImage(im);
    }

    template <class T>
    QtImageViewer<T>::~QtImageViewer()
    {
        close();

    #ifdef USE_QWT
        if (histoPlot)
          delete histoPlot;
        if (profilePlot)
          delete profilePlot;
    #endif // USE_QWT
    }


    template <class T>
    void QtImageViewer<T>::setImage(Image<T> &im)
    {
        ImageViewer<T>::setImage(im);
        BASE_QT_VIEWER::setImageSize(im.getWidth(), im.getHeight(), im.getDepth());
        if (im.getName()!=string(""))
          setName(this->image->getName());
        if (qOverlayImage)
        {
          delete qOverlayImage;
          qOverlayImage = NULL;
        }
    }

    template <class T>
    void QtImageViewer<T>::show()
    {
        if (!BASE_QT_VIEWER::isVisible())
        {
            drawImage();
            BASE_QT_VIEWER::dataChanged();
        }
        BASE_QT_VIEWER::showNormal();
        BASE_QT_VIEWER::raise();
        BASE_QT_VIEWER::activateWindow();
    }

    template <class T>
    void QtImageViewer<T>::showLabel()
    {
        this->setLabelImage(true);
        if (!BASE_QT_VIEWER::isVisible())
        {
            drawImage();
            BASE_QT_VIEWER::dataChanged();
        }
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
        if (profilePlot && profilePlot->isVisible())
          displayProfile(true);
    #endif // USE_QWT
        qApp->processEvents();
    }


    template <class T>
    void QtImageViewer<T>::saveSnapshot(const char *fileName)
    {
        BASE_QT_VIEWER::saveAs(fileName);
    }

    template <class T>
    void QtImageViewer<T>::drawImage()
    {
        typename Image<T>::sliceType lines = this->image->getSlices()[slider->value()];

        size_t w = this->image->getWidth();
        size_t h = this->image->getHeight();

        UINT8 *destLine;
        double coeff;
        double floor = ImDtTypes<T>::min();


        if (parentClass::labelImage)
          coeff = 1.0;
        else
        {
            if (autoRange)
            {
                vector<T> rangeV = rangeVal(*this->image);
                floor = rangeV[0];
                coeff = 255. / double(rangeV[1]-rangeV[0]);
            }
            else
              coeff = 255. / ( double(ImDtTypes<T>::max()) - double(ImDtTypes<T>::min()) );
        }

        for (size_t j=0;j<h;j++,lines++)
        {
            typename Image<T>::lineType pixels = *lines;

            destLine = this->qImage->scanLine(j);
            for (size_t i=0;i<w;i++)
    //           pixels[i] = 0;
                destLine[i] = (UINT8)(coeff * (double(pixels[i]) - floor));
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
    void QtImageViewer<T>::drawOverlay(const Image<T> &im)
    {
        if (!qOverlayImage)
          createOverlayImage();

        typename Image<T>::sliceType lines = im.getSlices()[slider->value()];
        typename Image<T>::lineType pixels;

        qOverlayImage->fill(0);

        QRgb *destLine;

        for (size_t j=0;j<im.getHeight();j++)
        {
            destLine = (QRgb*)(this->qOverlayImage->scanLine(j));
            pixels = *lines++;
            for (size_t i=0;i<im.getWidth();i++)
            {
              if (pixels[i]!=T(0))
                destLine[i] = overlayColorTable[(UINT8)pixels[i]];
            }
        }

        BASE_QT_VIEWER::overlayDataChanged();
        BASE_QT_VIEWER::update();
    }

    template <class T>
    RES_T QtImageViewer<T>::getOverlay(Image<T> &img)
    {
        if (!qOverlayImage)
          createOverlayImage();

        img.setSize(qOverlayImage->width(), qOverlayImage->height());

        QRgb *srcLine;
        int value;
        typename Image<T>::sliceType lines = img.getLines();
        typename Image<T>::lineType pixels;

        for (int j=0;j<qOverlayImage->height();j++)
        {
            srcLine = (QRgb*)(this->qOverlayImage->scanLine(j));
            pixels = *lines++;
            for (int i=0;i<qOverlayImage->width();i++)
            {
                value = overlayColorTable.indexOf(srcLine[i]);
                pixels[i] = value>=0 ? T(value) : T(0);
            }
        }
        img.modified();
        return RES_OK;
    }

    template <class T>
    void QtImageViewer<T>::overlayDataChanged(bool triggerEvents)
    {
        BASE_QT_VIEWER::overlayDataChanged();
        if (!triggerEvents)
          return;
        Event event(this);
        ImageViewer<T>::onOverlayModified.trigger(&event);
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
            baseColorTable.append(qRgb((*it).second[0], (*it).second[1], (*it).second[2]));
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
        if (qOverlayImage)
        {
            int index = overlayColorTable.indexOf(qOverlayImage->pixel(x,y));
            if (index>=0)
              txt = txt + "\n+ " + QString::number(index);
        }
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
        typename ImDtTypes<T>::lineType pLine = pSlice[0]; // dummy initialization
        T pVal;

        QGraphicsTextItem *textItem;
        QList<QGraphicsTextItem*>::Iterator txtIt = magnView->getTextItemList()->begin();

        QColor lightCol = QColor::fromRgb(255,255,255);
        QColor darkCol = QColor::fromRgb(0,0,0);
        T lightThresh = T(double(ImDtTypes<T>::max()-ImDtTypes<T>::min()) * 0.55);

        bool inYRange;
        
        for (int j=0;j<gridSize;j++,yi++)
        {
            if (yi>=0 && yi<imH)
            {
                inYRange = true;
                pLine = pSlice[yi];
            }
            else inYRange = false;

            for (int i=0,xi=x-gridSize/2; i<gridSize; i++,xi++)
            {
                textItem = *txtIt++;
                if (inYRange && xi>=0 && xi<imW)
                {
                    pVal = pLine[xi];
                    if (pVal<lightThresh)
                        textItem->setDefaultTextColor(lightCol);
                    else
                        textItem->setDefaultTextColor(darkCol);
                    textItem->setPlainText(ImDtTypes<T>::toString(pVal).c_str());
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
#ifdef Q_OS_WIN32
          read(urls[0].toString().remove("file:///").toStdString().c_str(), *this->image);
#else // Q_OS_WIN32
          read(urls[0].toString().remove("file:/").toStdString().c_str(), *this->image);
#endif // Q_OS_WIN32
        else
        {
            vector<string> files;
            for (QList<QUrl>::iterator it=urls.begin();it!=urls.end();it++)
#ifdef Q_OS_WIN32
              files.push_back((*it).toString().remove("file:///").toStdString());
#else // Q_OS_WIN32
              files.push_back((*it).toString().remove("file:/").toStdString());
#endif // Q_OS_WIN32
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
            histoPlot = new PlotWidget();
            histoPlot->setWindowTitle(QString(this->image->getName()) + " histogram");
            histoPlot->setAxisScale(QwtPlot::xBottom, ImDtTypes<T>::min(), ImDtTypes<T>::max());

            QwtPlotCurve *defaultCurve = histoPlot->getCurrentCurve();
            defaultCurve->setStyle( QwtPlotCurve::Steps );
            defaultCurve->setBrush( Qt::blue );
        }


        map<T, UINT> hist = histogram(*(this->image));

        QwtPlotCurve *curve = histoPlot->getCurrentCurve();

#if QWT_VERSION < 0x060000
        int ptsNbr = hist.size();
        double *xVals = new double[ptsNbr];
        double *yVals = new double[ptsNbr];

        int i=0;
        for(typename map<T,UINT>::iterator it=hist.begin();it!=hist.end();it++,i++)
        {
            xVals[i] = (*it).first;
            yVals[i] = (*it).second;
        }
        curve->setData(xVals, yVals, ptsNbr);

        histoPlot->replot();
        histoPlot->show();

        delete[] xVals;
        delete[] yVals;
#else // QWT_VERSION < 0x060000
        QVector<QPointF> samples;
        for(typename map<T,UINT>::iterator it=hist.begin();it!=hist.end();it++)
          samples.push_back(QPointF((*it).first, (*it).second));

        QwtPointSeriesData *myData = new QwtPointSeriesData();
        myData->setSamples(samples);
        curve->setData(myData);

        histoPlot->replot();
        histoPlot->show();
#endif // QWT_VERSION < 0x060000

    }

    template <class T>
    void QtImageViewer<T>::displayProfile(bool update)
    {
        if (update && (!profilePlot || !profilePlot->isVisible()))
          return;

        if (!update && profilePlot && profilePlot->isVisible())
        {
            profilePlot->raise();
            profilePlot->activateWindow();
            return;
        }

        if (!profilePlot)
        {
            profilePlot = new PlotWidget();
            profilePlot->setWindowTitle(QString(this->image->getName()) + " profile");
        }
//         profilePlot->detachItems();

        QwtPlotCurve *curve = profilePlot->getCurrentCurve();

        QLineF lnF(this->line->line());

        vector<IntPoint> bPoints = bresenhamPoints(lnF.x1(), lnF.y1(), lnF.x2(), lnF.y2());

        T value;
        typename Image<T>::sliceType lines = this->image->getSlices()[slider->value()];
        int i = 0;


#if QWT_VERSION < 0x060000
        int ptsNbr = bPoints.size();
        double *xVals = new double[ptsNbr];
        double *yVals = new double[ptsNbr];

        for(vector<IntPoint>::iterator it=bPoints.begin();it!=bPoints.end();it++,i++)
        {
            value = lines[(*it).y][(*it).x];
            xVals[i] = i;
            yVals[i] = value;
        }
        curve->setData(xVals, yVals, ptsNbr);

        profilePlot->replot();
        profilePlot->show();

        delete[] xVals;
        delete[] yVals;

#else // QWT_VERSION < 0x060000
        QVector<QPointF> samples;
        for(vector<IntPoint>::iterator it=bPoints.begin();it!=bPoints.end();it++,i++)
        {
            value = lines[(*it).y][(*it).x];
            samples.push_back(QPointF(i, value));
        }

        QwtPointSeriesData *myData = new QwtPointSeriesData();
        myData->setSamples(samples);
        curve->setData(myData);

        profilePlot->replot();
        profilePlot->show();
#endif // QWT_VERSION < 0x060000


    }
#endif // USE_QWT

} // namespace smil

#endif // _D_QT_IMAGE_VIEWER_HXX
