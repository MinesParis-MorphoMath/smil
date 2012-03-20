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


#ifndef _D_IMAGE_VIEWER_H
#define _D_IMAGE_VIEWER_H

#include "DImage.h"
#include "Qt/ImageViewerWidget.h"
#include "Qt/ImageViewer.h"
#include <QApplication>

template <class T>
class imageViewer : public baseImageViewer<T>
{
public:
    imageViewer();
    ~imageViewer();
    virtual void show();
    virtual void hide();
    virtual bool isVisible();
    virtual void setName(const char* name);
    virtual void loadFromData(typename Image<T>::lineType pixels, UINT w, UINT h);
    QApplication *_qapp;
protected:
    ImageViewerWidget *qtViewer;
//     ImageViewer *qtViewer;
};

template <class T>
imageViewer<T>::imageViewer()
{
    if (!qApp)
    {
        cout << "created" << endl;
        int ac = 1;
        char **av = NULL;
        _qapp = new QApplication(ac, av);
    }
    qtViewer = new ImageViewerWidget();
//     qtViewer = new ImageViewer();
}

template <class T>
imageViewer<T>::~imageViewer()
{
    hide();
    delete qtViewer;
}

template <class T>
void imageViewer<T>::show()
{
    qtViewer->show();
}

template <class T>
void imageViewer<T>::hide()
{
    qtViewer->hide();
}

template <class T>
bool imageViewer<T>::isVisible()
{
    return qtViewer->isVisible();
}

template <class T>
void imageViewer<T>::setName(const char* name)
{
    qtViewer->setName(name);
}

template <class T>
void imageViewer<T>::loadFromData(typename Image<T>::lineType pixels, UINT w, UINT h)
{
    cout << "Not implemented for this data type." << endl;
}

template <>
void imageViewer<UINT8>::loadFromData(Image<UINT8>::lineType pixels, UINT w, UINT h);

template <>
void imageViewer<UINT16>::loadFromData(Image<UINT16>::lineType pixels, UINT w, UINT h);

template <>
void imageViewer<Bit>::loadFromData(Image<Bit>::lineType pixels, UINT w, UINT h);

#endif // _D_IMAGE_VIEWER_H
