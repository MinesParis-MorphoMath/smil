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


#include "MagnifyView.h"

#include <QPainter>
#include <iostream>

MagnifyView::MagnifyView(QWidget *parent) :
        QGraphicsView(parent)
{
    fullImage = NULL;
    scaleFactor = 250;

    scene = NULL;

    textItemList = new QList<QGraphicsTextItem*>();

    setGridSize(7);
}

MagnifyView::~MagnifyView()
{
    delete scene;
    delete textItemList;
}

void MagnifyView::mouseMoveEvent(QMouseEvent *pEvent)
{
    // avoid to have the mouse blocked on me...
    hide();
    QGraphicsView::mouseMoveEvent(pEvent);
}

void MagnifyView::setImage(QImage *img)
{
    fullImage = img;
//     hide();
}

void MagnifyView::setGridSize(int s)
{
    gridSize = s;
    double cellSize = scaleFactor / gridSize;

    if (scene)
      delete scene;
    
    scene = new QGraphicsScene(this);
    setScene(scene);
    pixItem = scene->addPixmap(QPixmap());
    textItemList->clear();
    
    QPainterPath path;

    for (int j=0;j<gridSize;j++)
        for (int i=0;i<gridSize;i++)
        {
            path.addRect(i*cellSize, j*cellSize, cellSize, cellSize);
            QGraphicsTextItem *textItem = scene->addText(QString::number(i));
            textItem->setTextWidth(cellSize);
            textItem->setPos(i*cellSize, j*cellSize);
            textItemList->append(textItem);
        }
    int c = gridSize/2;
    pathItem = scene->addPath(path);
    
    QPainterPath path2;
    path2.addRect(c*cellSize+1, c*cellSize+1, cellSize-2, cellSize-2);
    centerRectPathItem = scene->addPath(path2);
    centerRectPathItem->setPen(QColor(255,0,0));
    
    adjustSize();
    resize(width()+1, height()+1); // adjusted size is sometimes to short...
}


void MagnifyView::displayAt(int x, int y)
{
    if (!fullImage)
        return;

    int xi = x-(gridSize-1)/2;
    int yi = y-(gridSize-1)/2;

    QImage img = fullImage->copy(xi, yi, gridSize, gridSize).scaled(scaleFactor, scaleFactor);
    pixItem->setPixmap(QPixmap::fromImage(img));

}

void MagnifyView::zoomIn()
{
    if (gridSize>20)
      return;
    scaleFactor *= double(gridSize+2)/double(gridSize);
    setGridSize(gridSize+2);
}

void MagnifyView::zoomOut()
{
    if (gridSize==1)
      return;
    scaleFactor *= double(gridSize-2)/double(gridSize);
    setGridSize(gridSize-2);
}

void MagnifyView::scaleImage(double /*factor*/)
{
}

