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
#include <QApplication>

#include "ImageViewerWidget.h"

#define RAND_UINT8 int(double(qrand())/RAND_MAX*255)

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
    qOverlayImage = new QImage();

    magnView = new MagnifyView(this);
    magnView->hide();
    magnActivated = false;

    valueLabel = new QLabel(this);
    valueLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    valueLabel->setAutoFillBackground(true);
    valueLabel->hide();
    valueLblActivated = true;

    imScene = new QImageGraphicsScene();
    imagePixmap = imScene->addPixmap( QPixmap() );
    overlayPixmap = imScene->addPixmap( QPixmap() );
    initColorTables();
    drawLabelized = false;
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
    delete qOverlayImage;
    delete imScene;
    delete magnView;

    delete zoomInAct;
    delete zoomOutAct;
}

void ImageViewerWidget::initColorTables()
{
    baseColorTable.clear();
    for (int i=0;i<256;i++)
      baseColorTable.append(qRgb(i, i, i));
    
    labelColorTable.clear();
//     qsrand(3);
    labelColorTable.append(qRgb(0, 0, 0));
    UINT8 r = 255, g = 0, b = 0;
    UINT8 minTh = 50;
    for (int i=0;i<3;i++)
    {
	r+=RAND_UINT8/2;
	g+=RAND_UINT8/2;
	b+=RAND_UINT8/2;
    }
    for (int i=1;i<256;i++)
    {	
	labelColorTable.append(qRgb(r+=RAND_UINT8/2, g+=RAND_UINT8/2, b+=RAND_UINT8/2));
	// Avoid to have both r, g and b to low (->black)
	while (r<minTh && g <minTh && b<minTh)
	{
	    r += RAND_UINT8/2;
	    g += RAND_UINT8/2;
	    b += RAND_UINT8/2;
	}
    }
    
    overlayColorTable.clear();
    overlayColorTable = labelColorTable;
}

void ImageViewerWidget::setLabelImage(bool val)
{
    if (drawLabelized==val)
      return;
    
    drawLabelized = val;
    if (drawLabelized)
      qImage->setColorTable(labelColorTable);
    else
      qImage->setColorTable(baseColorTable);
    imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
    QGraphicsView::update();
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
//     connect(this, SIGNAL(onDataChanged()), this, SLOT(update()));

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
    delete qOverlayImage;

    qImage = new QImage(w, h, QImage::Format_Indexed8);
    qOverlayImage = new QImage(w, h, QImage::Format_ARGB32_Premultiplied);
    // Clear overlay
    overlayPixmap->setPixmap(QPixmap::fromImage(*qOverlayImage));
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
    imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
//     repaint();
    update();
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

// void ImageViewerWidget::update()
// {
//     this->dataChanged();
//     QGraphicsView::update();
//     repaint();
//     qApp->processEvents();
// }


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
    case Qt::Key_L:
	setLabelImage(!drawLabelized); // Switch label mode
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

