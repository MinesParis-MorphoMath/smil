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


#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QMainWindow>
#include <QScrollArea>
#include <QGraphicsPixmapItem>
#include <QGraphicsTextItem>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "ImageViewerWidget.h"

namespace Ui {
    class ImageViewer;
}



class ImageViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageViewer(QWidget *parent = 0);
    ~ImageViewer();

    void setName(const char *name);
public slots:
    void load(const QString fileName);
    void loadFromData(const uchar *data, int w, int h);
    
private:
    Ui::ImageViewer *ui;
    QLabel *pixelPosLabel;
    QLabel *pixelValLabel;
    QLabel *scaleLabel;
    QScrollArea *scrollArea;

    ImageViewerWidget *graphicsView;

    void connectActions();

    QString name;
protected:

private slots:
    void viewMouseMoveEvent ( QMouseEvent * event );
    void displayPixelData(int x, int y, int pixVal, bool insideImage);
    void displayScaleFactor(double sf);
};

#endif // IMAGEVIEWER_H
