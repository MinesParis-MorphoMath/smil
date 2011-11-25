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

    QList<QGraphicsTextItem*> *textItemList;

    double scaleFactor;

    void mouseMoveEvent(QMouseEvent* pEvent);
signals:

public slots:
    void zoomIn();
    void zoomOut();
    void scaleImage(double factor);
    void displayAt(int x, int y);
    void setGridSize(int s);

    void setImage(QImage *img);
};

#endif // MagnifyView_H
