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
    hide();
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

    int imW = fullImage->width();
    int imH = fullImage->height();

    double s = scaleFactor / gridSize;

    uchar *pLine, pVal;
    QGraphicsTextItem *textItem;
    QList<QGraphicsTextItem*>::Iterator txtIt = textItemList->begin();

    for (int j=0;j<gridSize;j++,yi++)
    {
        if (yi>=0 && yi<imH)
            pLine = fullImage->scanLine(yi);
        else pLine = NULL;

        for (int i=0,xi=x-gridSize/2; i<gridSize; i++,xi++)
        {
            textItem = *txtIt++;
            if (pLine && xi>=0 && xi<imW)
            {
                pVal = pLine[xi];
                if (pVal<140)
                    textItem->setDefaultTextColor(QColor::fromRgb(255,255,255));
                else
                    textItem->setDefaultTextColor(QColor::fromRgb(0,0,0));
                textItem->setPlainText(QString::number(pVal));
            }
            else textItem->setPlainText("");

        }
    }

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

