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

void QImageGraphicsScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit(onMousePressed(event));
    QGraphicsScene::mousePressEvent(event);
}



ImageViewerWidget::ImageViewerWidget(QWidget *parent)
        : QGraphicsView(parent)
{
    setFrameShape(NoFrame);

    initColorTables();
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
    imagePixmap = imScene->addPixmap( QPixmap() );
    overlayPixmap = imScene->addPixmap( QPixmap() );
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
    
    rainbowColorTable.clear();
    
    rainbowColorTable.clear();
    for(int i=0;i<256;i++)
      rainbowColorTable.append(QColor::fromHsvF(double(i)/256., 1.0, 1.0).rgb());
    
    labelColorTable.clear();
    labelColorTable.append(qRgb(0,0,0));
    unsigned char curC = 0;
    for(int i=0;i<255;i++,curC+=47)
      labelColorTable.append(rainbowColorTable[curC]);
    
    overlayColorTable.clear();
    overlayColorTable = labelColorTable;
}

void ImageViewerWidget::setImageSize(int w, int h)
{
    if (w==qImage->width() && h==qImage->height())
        return;

    delete qImage;

    qImage = new QImage(w, h, QImage::Format_Indexed8);
    imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
    magnView->setImage(qImage);
    qImage->setColorTable(baseColorTable);
    
    // Todo : find a cleaner way...
    hide();
    show();
    
    // Clear overlay
    overlayPixmap->setPixmap(QPixmap());
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
    
    if (magnActivated && lastPixX>=0)
	displayMagnifyView(lastPixX, lastPixY);
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



void ImageViewerWidget::load(const QString fileName)
{
    qImage->load(fileName);
    magnView->setImage(qImage);

    emit onDataChanged();
}

void ImageViewerWidget::dataChanged()
{
    imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
//     repaint();
//     update();
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

    int dx = 10, dy = 20;
    int newX = p.x(), newY = p.y();

    if (newX+valueLabel->width() > width())
      newX -= dx + valueLabel->width();
    else 
      newX += dx;
    
    if (newY-valueLabel->height() < 0)
      newY += dy;
    else 
      newY -= dy;
    
    valueLabel->move(QPoint(newX, newY));
    magnView->move(p + QPoint(20, 20));
    QGraphicsView::mouseMoveEvent(event);
}

void ImageViewerWidget::mousePressEvent ( QMouseEvent * event )
{
    Qt::MouseButton btn = event->button();
    
    if (btn==Qt::LeftButton)
      setDragMode(QGraphicsView::ScrollHandDrag);
    
    QGraphicsView::mousePressEvent(event);
}

void ImageViewerWidget::mouseReleaseEvent ( QMouseEvent * event )
{
    Qt::MouseButton btn = event->button();
    
    if (btn==Qt::LeftButton)
      setDragMode(QGraphicsView::NoDrag);
    
    QGraphicsView::mousePressEvent(event);
}

void ImageViewerWidget::sceneMouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
    int x = int(event->scenePos().rx());
    int y = int(event->scenePos().ry());

    UINT w = qImage->width();
    UINT h = qImage->height();
    
    if (x>=0 && x<w && y>=0 && y<h)
    {
        if (valueLblActivated)
	{
            valueLabel->show();
	    displayPixelValue(x, y);
	}
        if (magnActivated)
	{
	    displayMagnifyView(x, y);
            magnView->show();
	}
	lastPixX = x;
	lastPixY = y;
    }
    else
    {
        valueLabel->hide();
        magnView->hide();
	lastPixX = -1;
	lastPixY = -1;
    }
    
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
        if (magnActivated && lastPixX>=0) 
	{
	    displayMagnifyView(lastPixX, lastPixY);
	    magnView->show();
	}
        else magnView->hide();
        break;
    case Qt::Key_V:
        valueLblActivated = !valueLblActivated;
        if (valueLblActivated && lastPixX>=0) 
	    valueLabel->show();
        else valueLabel->hide();
        break;
    case Qt::Key_L:
	setLabelImage(!drawLabelized); // Switch label mode
	break;
    }

    emit onKeyPressEvent(event);
}


