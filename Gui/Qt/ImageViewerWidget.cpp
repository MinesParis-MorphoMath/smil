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
    qImage = new QImage();

    magnView = new MagnifyView(this);
    magnView->hide();
    magnActivated = false;

    valueLabel = new QLabel(this);
    valueLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    valueLabel->setAutoFillBackground(true);
    valueLabel->hide();
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
    delete qImage;
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
    if (w==qImage->width() && h==qImage->height())
        return;

    delete qImage;

    qImage = new QImage(w, h, QImage::Format_ARGB32);

//      qImage->setNumColors(512);
//     for (int i=0; i<256; i++)
//         qImage->setColor(i,qRgb(i,i,i));
//     qImage->setColorTable();
}


void ImageViewerWidget::load(const QString fileName)
{
    qImage->load(fileName);
    magnView->setImage(qImage);

    emit onDataChanged();
}

void ImageViewerWidget::dataChanged()
{
    magnView->setImage(qImage);
    repaint();
//     qApp->processEvents();
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
    pixItem->setPixmap(QPixmap::fromImage(*qImage));
//     imScene->setSceneRect(0, 0, qImage->width(), qImage->height());
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
  QImage ovIm(qImage->width(), qImage->height(), QImage::Format_ARGB32);
//   qImage->fill(Qt::transparent);
  ovIm.fill(Qt::transparent);
//   pit->setShapeMode(QGraphicsPixmapItem::MaskShape);
  
//   qImage->setColor(280,qRgb(255,0,0));
  for (int i=0;i<qImage->width();i++)
    ovIm.setPixel(i,10, 256);
  
//   QGraphicsPixmapItem *pit = imScene->addPixmap(QPixmap::fromImage(ovIm));
  
  qImage->setAlphaChannel(ovIm);
  
  this->repaint();
  this->dataChanged();
  
  return;
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

    if (x>=0 && x<qImage->width() && y>=0 && y<qImage->height())
    {
        magnView->displayAt(x, y);
        if (valueLblActivated)
            valueLabel->show();
        if (magnActivated)
            magnView->show();
	imageMouseMoveEvent(event);
    }
    else
    {
        valueLabel->adjustSize();
        valueLabel->hide();
        magnView->hide();
    }
    
}

