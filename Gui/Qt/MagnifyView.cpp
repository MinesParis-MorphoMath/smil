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


#include "MagnifyView.h"

#include <QPainter>
#include <iostream>

MagnifyView::MagnifyView(QWidget *parent) :
        QGraphicsView(parent)
{
    fullImage = NULL;
    scaleFactor = 250;

    scene = new QGraphicsScene();
    pixItem = scene->addPixmap(QPixmap());
    pathItem = scene->addPath(QPainterPath());
    setScene(scene);

    textItemList = new QList<QGraphicsTextItem*>();

    setGridSize(7);

    hide();
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

    for (int i=0;i<textItemList->count();i++)
    {
        QGraphicsTextItem *item = textItemList->first();
        textItemList->pop_front();
        scene->removeItem(item);
    }

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
    path.addRect(c*cellSize+1, c*cellSize+1, cellSize-2, cellSize-2);
    pathItem->setPath(path);
}


void MagnifyView::displayAt(int x, int y)
{
    if (!fullImage)
        return;

    int xi = x-gridSize/2;
    int yi = y-gridSize/2;

    QImage img = fullImage->copy(xi, yi, gridSize, gridSize).scaled(scaleFactor, scaleFactor);
    QPixmap px = QPixmap::fromImage(img);
    pixItem->setPixmap(px);

}

void MagnifyView::zoomIn()
{
    scaleImage(1.25);
}

void MagnifyView::zoomOut()
{
    scaleImage(0.8);
}

void MagnifyView::scaleImage(double factor)
{
//     Q_ASSERT(pixmap());
//     scaleFactor *= factor;
//     resize(scaleFactor * pixmap()->size());
}

