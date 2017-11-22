/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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

#ifndef MagnifyView_H
#define MagnifyView_H

#include <QLabel>
#include <QMouseEvent>
#include <QAction>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsTextItem>
#include <QGraphicsScene>
#include <QGraphicsPathItem>

#include "Core/include/DCommon.h"

class MagnifyView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit MagnifyView(QWidget *parent = 0);

    ~MagnifyView();

private:
    QImage *fullImage;

    int gridSize;
    QGraphicsScene *scene;
    QGraphicsPixmapItem *pixItem;
    QGraphicsPathItem *pathItem;
    QGraphicsPathItem *centerRectPathItem;

    QList<QGraphicsTextItem*> *textItemList;

    double scaleFactor;

    void mouseMoveEvent(QMouseEvent* pEvent);
    
public:
    void setGridSize(int s);
    inline int getGridSize() { return gridSize; }
    inline double getScaleFactor() { return scaleFactor; }
    inline QGraphicsPixmapItem *getPixItem() { return pixItem; }
    inline QList<QGraphicsTextItem*> *getTextItemList() { return textItemList; }
    inline QGraphicsPathItem *getCenterRectPathItem() { return centerRectPathItem; }
    void displayAt(int x, int y);
    void setImage(QImage *img);

public slots:
    void zoomIn();
    void zoomOut();
    void scaleImage(double factor);
};

#endif // MagnifyView_H
