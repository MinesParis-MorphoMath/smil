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


#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QTimer>

#include "ImageViewerWidget.h"



QImageGraphicsScene::QImageGraphicsScene(QObject *parent)
        : QGraphicsScene(parent)
{
}

void QImageGraphicsScene::mouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
    emit(onMouseMove(event));
    QGraphicsScene::mouseMoveEvent(event);
}



ImageViewerWidget::ImageViewerWidget(QWidget *parent)
        : QGraphicsView(parent)
{
    setFrameShape(NoFrame);

    scaleFactor = 1.0;
    image = new QImage();

    magnView = new MagnifyView(this);
    magnView->hide();
    magnActivated = false;

    valueLabel = new QLabel(this);
    valueLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    valueLabel->setAutoFillBackground(true);
//     valueLabel->hide();
    valueLblActivated = true;

    imScene = new QImageGraphicsScene();
    pixItem = imScene->addPixmap( QPixmap() );
    this->setScene( imScene );

    setMouseTracking(true);
//     this->grabKeyboard();

    update();
    scale(1.0);

    createActions();
    connectActions();
}

ImageViewerWidget::~ImageViewerWidget()
{
    delete image;
    delete imScene;
    delete magnView;

    delete zoomInAct;
    delete zoomOutAct;
}


void ImageViewerWidget::createActions()
{
    zoomInAct = new QAction(tr("&Zoom in"), this);
    zoomInAct->setShortcut(tr("z"));
    zoomInAct->setEnabled(true);

    zoomOutAct = new QAction(tr("Zoom out"), this);
    zoomOutAct->setShortcut(tr("a"));
}

void ImageViewerWidget::connectActions()
{
    connect(this, SIGNAL(onDataChanged()), this, SLOT(update()));

    connect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));

    connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
    connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));

//    connect(ui->zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
//    connect(ui->zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));
}

void ImageViewerWidget::setName(const char *new_name)
{
    name = new_name;
    updateTitle();
}

void ImageViewerWidget::updateTitle()
{
    setWindowTitle(name);
}

void ImageViewerWidget::setImageSize(int w, int h)
{
    if (w==image->width() && h==image->height())
        return;

    delete image;

    image = new QImage(QSize(w, h), QImage::Format_Indexed8);

    image->setNumColors(256);
    for (int i=0; i<256; i++)
        image->setColor(i,qRgb(i,i,i));
}


void ImageViewerWidget::load(const QString fileName)
{
    image->load(fileName);
    magnView->setImage(image);

    emit onDataChanged();
}

void ImageViewerWidget::loadFromData(const uchar *data, int w, int h)
{
    setImageSize(w, h);

    for (int j=0;j<h;j++)
        memcpy(image->scanLine(j), data+(j*w), sizeof(uchar) * w);

    magnView->setImage(image);

    emit onDataChanged();
}


void ImageViewerWidget::zoomIn()
{
    scale(1.25);
}

void ImageViewerWidget::zoomOut()
{
    scale(0.8);
}


void ImageViewerWidget::scale(double factor)
{
    scaleFactor *= factor;
    QGraphicsView::scale(factor, factor);

    emit(onRescaled(scaleFactor));
}

void ImageViewerWidget::update()
{
    pixItem->setPixmap(QPixmap::fromImage(*image));
//     imScene->setSceneRect(0, 0, image->width(), image->height());
}


void ImageViewerWidget::mouseMoveEvent ( QMouseEvent * event )
{
    QPoint p = event->pos();

    int dx = 40, dy = -20;
    magnView->move(p + QPoint(dx, dy));
    valueLabel->move(p + QPoint(dx, dy-valueLabel->height()));

    QGraphicsView::mouseMoveEvent(event);
}

void ImageViewerWidget::wheelEvent ( QWheelEvent * event )
{
    if (event->delta()>0)
        zoomIn();
    else zoomOut();

    QGraphicsView::wheelEvent(event);
}

void ImageViewerWidget::keyPressEvent(QKeyEvent *event)
{
    QGraphicsView::keyPressEvent(event);

    switch (event->key())
    {
    case Qt::Key_Z:
        zoomIn();
        break;
    case Qt::Key_A:
        zoomOut();
        break;
    case Qt::Key_M:
        magnActivated = !magnActivated;
        if (magnActivated) magnView->show();
        else magnView->hide();
        break;
    case Qt::Key_V:
        valueLblActivated = !valueLblActivated;
        if (valueLblActivated) valueLabel->show();
        else valueLabel->hide();
        break;
    }

    emit onKeyPressEvent(event);
}

#include <iostream>

void ImageViewerWidget::sceneMouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
    int x = int(event->scenePos().rx());
    int y = int(event->scenePos().ry());
    bool isOnImage;
    int pixVal = -1;

    if (x>=0 && x<image->width() && y>=0 && y<image->height())
    {
        isOnImage = true;
        pixVal = image->scanLine(y)[x];
        valueLabel->setText("(" + QString::number(x) + ", " + QString::number(y) + ") " + QString::number(pixVal));
        valueLabel->adjustSize();
        magnView->displayAt(x, y);
        if (valueLblActivated)
            valueLabel->show();
        if (magnActivated)
            magnView->show();
    }
    else
    {
        isOnImage = false;
        valueLabel->adjustSize();
        valueLabel->hide();
        magnView->hide();
    }

    emit(onCursorPixelValueChanged(x, y, pixVal, isOnImage));
}

