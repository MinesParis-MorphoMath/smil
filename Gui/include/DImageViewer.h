/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef _D_IMAGE_VIEWER_H
#define _D_IMAGE_VIEWER_H

#include "DImage.h"
#include "Qt/ImageViewerWidget.h"
#include "Qt/ImageViewer.h"
#include <QApplication>

class imageViewer : public baseImageViewer
{
public:
    imageViewer();
    ~imageViewer();
    virtual inline void show();
    virtual inline bool isVisible();
    virtual inline void setName(const char* name);
    virtual inline void loadFromData(void *pixels, UINT w, UINT h);
    QApplication *_qapp;
private:
    ImageViewerWidget *qtViewer;
//     ImageViewer *qtViewer;
};

#endif // _D_IMAGE_VIEWER_H
