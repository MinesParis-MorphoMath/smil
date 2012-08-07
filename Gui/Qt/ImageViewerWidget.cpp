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


#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QTimer>
#include <QApplication>

#include "ImageViewerWidget.h"

#define RAND_UINT8 int(double(qrand())/RAND_MAX*255)


#define PIXMAP_MAX_DIM 32767

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
    // Allows to zoom under the mouse pixel
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    initColorTables();
    scaleFactor = 1.0;
    qImage = new QImage();
    qOverlayImage = NULL;

    magnView = new MagnifyView(this);
    magnView->hide();
    magnActivated = false;

    valueLabel = new QLabel(this);
    valueLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    valueLabel->setAutoFillBackground(true);
    valueLabel->hide();
    valueLblActivated = true;

    hintLabel = new QLabel(this);
    hintLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    hintLabel->setAutoFillBackground(true);
    hintLabel->hide();
    hintLabel->move(QPoint(10,10));
    hintLabel->setText("ok");
    hintLabel->setEnabled(false);
    
    hintTimer = new QTimer();
    hintTimer->setSingleShot(true);
    connect( hintTimer, SIGNAL(timeout()), hintLabel, SLOT(hide()) );
    
    imagePixmaps.clear();
    overlayPixmaps.clear();
    
    imScene = NULL;
    drawLabelized = false;

    setMouseTracking(true);
//     this->grabKeyboard();

    update();
    scale(1.0);

    createActions();
    connectActions();
    
    hintLabel->setEnabled(true);
    
    layout = new QGridLayout(this);
    layout->setAlignment(Qt::AlignBottom);
    
    slider = new QSlider(Qt::Horizontal, this);
    slider->setSliderPosition(0);
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)) );
    slider->hide();
    layout->addWidget(slider);
    
}

ImageViewerWidget::~ImageViewerWidget()
{
    delete slider;
    delete layout;
    
    delete qImage;
    if (qOverlayImage)
      delete qOverlayImage;
    delete imScene;
    delete magnView;
    
    delete valueLabel;
    delete hintLabel;
    delete hintTimer;

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

void ImageViewerWidget::setImageSize(int w, int h, int d)
{
    if (d<=slider->sliderPosition())
      slider->setSliderPosition(d-1);
    if (d>1)
    {
	slider->setMaximum(d-1);
	slider->show();
    }
    else slider->hide();
    
    if (w==qImage->width() && h==qImage->height())
        return;

    delete qImage;
    
    qImage = new QImage(w, h, QImage::Format_Indexed8);
    
    
    // Create tiled pixmaps
    
    if (imScene)
      delete imScene;
    imScene = new QImageGraphicsScene();
    setScene(imScene);
    connect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));
    
    imagePixmaps.clear();
    
    UINT pixNbrX = w/PIXMAP_MAX_DIM + 1;
    UINT pixNbrY = h/PIXMAP_MAX_DIM + 1;
    UINT pixW = PIXMAP_MAX_DIM, pixH = PIXMAP_MAX_DIM;
    
    for (int j=0;j<pixNbrY;j++)
    {
	if (j==pixNbrY-1)
	  pixH = h%PIXMAP_MAX_DIM;
	
	for (int i=0;i<pixNbrX;i++)
	{
	    if (i==pixNbrX-1)
	      pixW = w%PIXMAP_MAX_DIM;
	    
	  QGraphicsPixmapItem *item = imScene->addPixmap(QPixmap(pixW, pixH));
	  item->moveBy(i*PIXMAP_MAX_DIM, j*PIXMAP_MAX_DIM);
	  imagePixmaps.append(item);
	  
	  item = imScene->addPixmap(QPixmap());
	  item->moveBy(i*PIXMAP_MAX_DIM, j*PIXMAP_MAX_DIM);
	  overlayPixmaps.append(item);
	}
    }
    
//     imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
//     QImage im = qImage->copy(50,0,200,200);

    //     magnView->setImage(qImage);
    qImage->setColorTable(baseColorTable);
    
    // Clear overlay
//     overlayPixmap->setPixmap(QPixmap());
    
//     resize(imScene->width(), imScene->height());
    adjustSize();
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
//     imagePixmap->setPixmap(QPixmap::fromImage(*qImage));
    
    if (magnActivated && lastPixX>=0)
	displayMagnifyView();
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

//     connect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));

    connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
    connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));

//    connect(ui->zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
//    connect(ui->zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));
}

void ImageViewerWidget::setName(QString new_name)
{
    name = new_name;
    updateTitle();
}

void ImageViewerWidget::updateTitle()
{
    setWindowTitle(name);
}

void ImageViewerWidget::displayHint(QString msg)
{
    if (!hintLabel->isEnabled())
      return;
    
    hintLabel->setText(msg);
    hintLabel->show();
    hintTimer->start(1000);
}

void ImageViewerWidget::load(const QString fileName)
{
    qImage->load(fileName);
    magnView->setImage(qImage);

    emit onDataChanged();
}

void ImageViewerWidget::dataChanged()
{
    UINT w = qImage->width(), h = qImage->height();
    
    UINT pixNbrX = w/PIXMAP_MAX_DIM + 1;
    UINT pixNbrY = h/PIXMAP_MAX_DIM + 1;
    UINT pixW = PIXMAP_MAX_DIM, pixH = PIXMAP_MAX_DIM;
    
    QList<QGraphicsPixmapItem*>::iterator it = imagePixmaps.begin();
    
    for (int j=0;j<pixNbrY;j++)
    {
	if (j==pixNbrY-1)
	  pixH = h%PIXMAP_MAX_DIM;
	
	for (int i=0;i<pixNbrX;i++)
	{
	    if (i==pixNbrX-1)
	      pixW = w%PIXMAP_MAX_DIM;
	    
	    QGraphicsPixmapItem *item = *it;
	    item->setPixmap(QPixmap::fromImage(qImage->copy(i*PIXMAP_MAX_DIM, j*PIXMAP_MAX_DIM, pixW, pixH)));
	    it++;
	}
    }
  
  
    magnView->setImage(qImage);
    
    emit onDataChanged();
}

void ImageViewerWidget::overlayDataChanged()
{
    UINT w = qImage->width(), h = qImage->height();
    
    UINT pixNbrX = w/PIXMAP_MAX_DIM + 1;
    UINT pixNbrY = h/PIXMAP_MAX_DIM + 1;
    UINT pixW = PIXMAP_MAX_DIM, pixH = PIXMAP_MAX_DIM;
    
    QList<QGraphicsPixmapItem*>::iterator it = overlayPixmaps.begin();
    
    for (int j=0;j<pixNbrY;j++)
    {
	if (j==pixNbrY-1)
	  pixH = h%PIXMAP_MAX_DIM;
	
	for (int i=0;i<pixNbrX;i++)
	{
	    if (i==pixNbrX-1)
	      pixW = w%PIXMAP_MAX_DIM;
	    
	    QGraphicsPixmapItem *item = *it;
	    item->setPixmap(QPixmap::fromImage(qOverlayImage->copy(i*PIXMAP_MAX_DIM, j*PIXMAP_MAX_DIM, pixW, pixH)));
	    it++;
	}
    }
}



void ImageViewerWidget::clearOverlay()
{
    QList<QGraphicsPixmapItem*>::iterator it = overlayPixmaps.begin();

    while(it!=overlayPixmaps.end())
    {
	(*it)->setPixmap(QPixmap());
	it++;
    }
    
    delete qOverlayImage;
    qOverlayImage = NULL;

    update();
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

    displayHint(QString::number(int(scaleFactor*100)) + "%");
    emit(onRescaled(scaleFactor));
}

void ImageViewerWidget::leaveEvent (QEvent *event)
{
    // Hide valueLabel and magnView when mouse quits the window
    valueLabel->hide();
    magnView->hide();
}

void ImageViewerWidget::mouseMoveEvent ( QMouseEvent * event )
{
    QPoint p = event->pos();

    int dx = 20, dy = 20;
    int newVlX = p.x(), newVlY = p.y();
    
    if (newVlX+valueLabel->width()+dx > width())
      newVlX -= dx + valueLabel->width();
    else 
      newVlX += dx;
    
    if (newVlY-dy < 0)
      newVlY += dy;
    else 
      newVlY -= dy;
    valueLabel->move(QPoint(newVlX, newVlY));
    
    
    int newMvX = p.x(), newMvY = p.y();
    
    if (newMvX+magnView->width()+dx > width()  &&  newMvX-dx-magnView->width()>=0)
      newMvX -= dx + magnView->width();
    else 
      newMvX += dx;
    
    if (newMvY+magnView->height() > height()  &&  newMvY-magnView->height()>=0)
      newMvY -= magnView->height();
    
    magnView->move(QPoint(newMvX, newMvY));
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
    if (event->modifiers() & Qt::ControlModifier)
    {
	if (event->delta()>0)
	    zoomIn();
	else zoomOut();
	
	return;
    }

    QGraphicsView::wheelEvent(event);
}

void ImageViewerWidget::keyPressEvent(QKeyEvent *event)
{
    QGraphicsView::keyPressEvent(event);

    switch (event->key())
    {
    case Qt::Key_Z:
	if (magnView->isVisible())
	{
	    magnView->zoomIn();
	    displayMagnifyView();
	}
	else zoomIn();
        break;
    case Qt::Key_A:
	if (magnView->isVisible())
	{
	    magnView->zoomOut();
	    displayMagnifyView();
	}
	else zoomOut();
        break;
    case Qt::Key_M:
        magnActivated = !magnActivated;
        if (magnActivated && lastPixX>=0) 
	{
	    displayMagnifyView();
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


