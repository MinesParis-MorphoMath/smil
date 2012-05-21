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

#include "DImageViewer.hpp"
#include "DTypes.h"

#include "Qt/ImageViewerWidget.h"
#include "Qt/ImageViewer.h"
#include <QApplication>

template <class T> class Image;

template <class T>
class qtImageViewer : public imageViewer<T>
{
public:
    typedef imageViewer<T> parentClass;
    qtImageViewer(Image<T> *im);
    ~qtImageViewer();
    virtual void show();
    virtual void hide();
    virtual bool isVisible();
    virtual void setName(const char* _name);
    virtual void loadFromData(typename ImDtTypes<T>::lineType pixels, UINT w, UINT h);
    QApplication *_qapp;
protected:
    ImageViewerWidget *qtViewer;
//     ImageViewer *qtViewer;
};


template <class T>
qtImageViewer<T>::qtImageViewer(Image<T> *im)
  : imageViewer<T>(im)
{
    if (!qApp)
    {
        cout << "created" << endl;
        int ac = 1;
        char **av = NULL;
        _qapp = new QApplication(ac, av);
    }
    qtViewer = new ImageViewerWidget();
    if (this->image->getName())
      setName(this->image->getName());
//     qtViewer = new ImageViewer();
}

template <class T>
qtImageViewer<T>::~qtImageViewer()
{
    hide();
    delete qtViewer;
}

template <class T>
void qtImageViewer<T>::show()
{
    if (qtViewer->isVisible())
      return;
    
    qtViewer->show();
    qtViewer->repaint();
    qApp->processEvents();
}

template <class T>
void qtImageViewer<T>::hide()
{
    qtViewer->hide();
}

template <class T>
bool qtImageViewer<T>::isVisible()
{
    return qtViewer->isVisible();
}

template <class T>
void qtImageViewer<T>::setName(const char* _name)
{
    parentClass::setName(_name);
    qtViewer->setName(_name);
}

template <class T>
void qtImageViewer<T>::loadFromData(typename ImDtTypes<T>::lineType pixels, UINT w, UINT h)
{
    cout << "Not implemented for this data type." << endl;
}

template <>
void qtImageViewer<UINT8>::loadFromData(ImDtTypes<UINT8>::lineType pixels, UINT w, UINT h);

template <>
void qtImageViewer<UINT16>::loadFromData(ImDtTypes<UINT16>::lineType pixels, UINT w, UINT h);

#ifdef SMIL_WRAP_Bit
template <>
void qtImageViewer<Bit>::loadFromData(ImDtTypes<Bit>::lineType pixels, UINT w, UINT h);
#endif // SMIL_WRAP_Bit


#endif // _D_QT_IMAGE_VIEWER_HPP
