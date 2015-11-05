/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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
#include <QMenu>
#include <QMimeData>
#include <QScrollBar>
#include <QColorDialog>
#include <QInputDialog>
#include <QFileDialog>

#ifdef USE_QWT
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#endif // USE_QWT


#include <math.h>

#include "ImageViewerWidget.h"
#include "ColorPicker.h"
#include "Gui/include/DGuiInstance.h"

#define RAND_UINT8 int(double(qrand())/RAND_MAX*255)


#define PIXMAP_MAX_DIM 2047

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
    emit(onMousePress(event));
    QGraphicsScene::mousePressEvent(event);
}

void QImageGraphicsScene::mouseReleaseEvent ( QGraphicsSceneMouseEvent * event )
{
    emit(onMouseRelease(event));
    QGraphicsScene::mouseReleaseEvent(event);
}



ImageViewerWidget::ImageViewerWidget(QWidget *parent)
        : QGraphicsView(parent)
{
    setFrameShape(NoFrame);
    // Allows to zoom under the mouse pixel
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setAcceptDrops(true);
    setContextMenuPolicy(Qt::CustomContextMenu);
        
    imageFormat = QImage::Format_Indexed8;
    
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

    hintLabel = new QLabel(this);
    hintLabel->setFrameStyle(QFrame::Panel | QFrame::Raised);
    hintLabel->setAutoFillBackground(true);
    hintLabel->hide();
    hintLabel->move(QPoint(10,10));
    hintLabel->setEnabled(true);
    
    hintTimer = new QTimer();
    hintTimer->setSingleShot(true);
    connect( hintTimer, SIGNAL(timeout()), hintLabel, SLOT(hide()) );
    
    iconTimer = new QTimer();
    iconTimer->setSingleShot(true);
    connect( iconTimer, SIGNAL(timeout()), this, SLOT(updateIcon()) );
    
    imagePixmaps.clear();
    overlayPixmaps.clear();
    
    imScene = NULL;
    drawLabelized = false;
    autoRange = false;

    setMouseTracking(true);
//     this->grabKeyboard();

    update();
    scale(1.0);

    createActions();
    
    connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    
    layout = new QGridLayout(this);
    layout->setAlignment(Qt::AlignBottom);
    
    slider = new QSlider(Qt::Horizontal, this);
    slider->setSliderPosition(0);
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)) );
    slider->hide();
    layout->addWidget(slider);
    
    colorPicker = new ColorPicker(this);
    colorPicker->setColors(overlayColorTable);
    connect(colorPicker, SIGNAL(colorChanged(const QColor &)), this, SLOT(setDrawPenColor(const QColor &)));
    
    cursorMode = cursorDraw;    
    line = new QGraphicsLineItem();
    line->setPen(QPen(Qt::blue, 1));
    drawing = false;
    
    drawPen.setColor(overlayColorTable[1]);
    drawPen.setWidth(2);
    
    setCursorMode(cursorMove);
}

ImageViewerWidget::~ImageViewerWidget()
{
    delete slider;
    delete layout;
    
    delete qImage;
    if (!qOverlayImage.isEmpty())
      deleteOverlayImage();
    delete imScene;
    delete magnView;
    
    delete valueLabel;
    delete hintLabel;
    delete hintTimer;

    // Delete actions
    for (QMap<QString, QAction*>::iterator it=actionMap.begin();it!=actionMap.end();it++)
      delete it.value();
    
    delete colorPicker;
}

void ImageViewerWidget::createOverlayImage()
{
    if (!qOverlayImage.isEmpty())
      deleteOverlayImage();
    for (size_t i=0;i<imDepth;i++)
    {
        QImage *im = new QImage(imWidth, imHeight, QImage::Format_ARGB32_Premultiplied);
        im->setColorTable(overlayColorTable);
        im->fill(Qt::transparent);
        qOverlayImage.append(im);
    }
}

void ImageViewerWidget::deleteOverlayImage()
{
    if (!qOverlayImage.isEmpty())
      return;
    for (size_t i=0;i<imDepth;i++)
      delete qOverlayImage[i];
    qOverlayImage.clear();
}

void ImageViewerWidget::updateIcon()
{
    int size = min(qImage->width(), qImage->height());
    setWindowIcon(QIcon(imagePixmaps[0]->pixmap().copy(0,0,size,size)));
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
    imWidth = w;
    imHeight = h;
    imDepth = d;
    
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
    
    qImage = new QImage(w, h, imageFormat);
    
    
    // Create tiled pixmaps
    
    if (imScene)
      delete imScene;
    imScene = new QImageGraphicsScene();
    setScene(imScene);
    connect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));
    connect(imScene, SIGNAL(onMousePress(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMousePressEvent(QGraphicsSceneMouseEvent*)));
    connect(imScene, SIGNAL(onMouseRelease(QGraphicsSceneMouseEvent*)), this, SLOT(sceneMouseReleaseEvent(QGraphicsSceneMouseEvent*)));
    
    imagePixmaps.clear();
    overlayPixmaps.clear();
    
    size_t pixNbrX = w/PIXMAP_MAX_DIM + 1;
    size_t pixNbrY = h/PIXMAP_MAX_DIM + 1;
    size_t pixW = PIXMAP_MAX_DIM, pixH = PIXMAP_MAX_DIM;
    
    for (size_t j=0;j<pixNbrY;j++)
    {
        if (j==pixNbrY-1)
          pixH = h%PIXMAP_MAX_DIM;
        
        pixW = PIXMAP_MAX_DIM;
        
        for (size_t i=0;i<pixNbrX;i++)
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
    

    //     magnView->setImage(qImage);
    qImage->setColorTable(baseColorTable);
    
    // Clear overlay
//     overlayPixmap->setPixmap(QPixmap());
    
    double minSize = 256;
    if (scaleFactor==1 && imScene->height()<minSize)
    {
        int scaleFact = log(minSize/imScene->height())/log(1.25);
        scale(pow(1.25, scaleFact), true);
    }
    adjustSize();
    if (scaleFactor==1 && QWidget::height()<qImage->height())
    {
        int scaleFact = log(double(QWidget::height())/qImage->height())/log(0.8);
        scale(pow(0.8, scaleFact), true);
    }
    if (!qOverlayImage.isEmpty())
      deleteOverlayImage();
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
    QAction *act;
    
    actionMap["zoomIn"] = act = new QAction("&Zoom in", this);
    act->setShortcut(tr("z"));
    connect(act, SIGNAL(triggered()), this, SLOT(zoomIn()));
    
    actionMap["zoomOut"] = act = new QAction("Zoom out", this);
    act->setShortcut(tr("a"));
    connect(act, SIGNAL(triggered()), this, SLOT(zoomOut()));
    
    actionMap["help"] = act = new QAction("Help", this);
    act->setShortcut(Qt::Key_F1);
    connect(act, SIGNAL(triggered()), this, SLOT(showHelp()));
    addAction(act);
    
    actionMap["saveAs"] = act = new QAction("Save snapshot", this);
    act->setShortcut(Qt::Key_S | Qt::CTRL);
    connect(act, SIGNAL(triggered()), this, SLOT(saveAs()));
    addAction(act);
}

void ImageViewerWidget::saveAs(const char *fileName)
{
    QString fName;
    if (fileName!=NULL)
      fName = fileName;
    else
      fName = QFileDialog::getSaveFileName(this, "Save Image", "", tr("Image Files (*.png *.jpg *.bmp *.tif)"));
    
    if (fName.isEmpty())
      return;
    QPixmap pixMap = QPixmap::grabWidget(this);  
    pixMap.save(fName);  
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

void ImageViewerWidget::displayHint(QString msg, int timerVal)
{
    if (!hintLabel->isEnabled())
      return;
    
    hintLabel->setText(msg);
    hintLabel->adjustSize();
    hintLabel->show();
    hintTimer->start(timerVal);
}

void ImageViewerWidget::load(const QString fileName)
{
    qImage->load(fileName);
    magnView->setImage(qImage);

    emit onDataChanged();
}

void ImageViewerWidget::updatePixmaps(QImage *image, QList<QGraphicsPixmapItem*> *pixmaps)
{
    size_t w = image->width(), h = image->height();
    
    size_t pixNbrX = w/PIXMAP_MAX_DIM + 1;
    size_t pixNbrY = h/PIXMAP_MAX_DIM + 1;
    size_t pixW = PIXMAP_MAX_DIM, pixH = PIXMAP_MAX_DIM;
    
    QList<QGraphicsPixmapItem*>::iterator it = pixmaps->begin();
    
    for (size_t j=0;j<pixNbrY;j++)
    {
        if (j==pixNbrY-1)
          pixH = h%PIXMAP_MAX_DIM;
        
        pixW = PIXMAP_MAX_DIM;
        
        for (size_t i=0;i<pixNbrX;i++)
        {
            if (i==pixNbrX-1)
              pixW = w%PIXMAP_MAX_DIM;
            
            QGraphicsPixmapItem *item = *it;
            item->setPixmap(QPixmap::fromImage(image->copy(i*PIXMAP_MAX_DIM, j*PIXMAP_MAX_DIM, pixW, pixH)));
            it++;
        }
    }
}

void ImageViewerWidget::dataChanged()
{
    updatePixmaps(qImage, &imagePixmaps);
  
    magnView->setImage(qImage);
    iconTimer->start(200);
    
    emit onDataChanged();
}

void ImageViewerWidget::overlayDataChanged(bool /*triggerEvents*/)
{
    updatePixmaps(qOverlayImage[slider->sliderPosition()], &overlayPixmaps);
}

void ImageViewerWidget::clearOverlay()
{
    if (qOverlayImage.isEmpty())
      return;
    
    QList<QGraphicsPixmapItem*>::iterator it = overlayPixmaps.begin();

    while(it!=overlayPixmaps.end())
    {
        (*it)->setPixmap(QPixmap());
        it++;
    }
    
    for (size_t i=0;i<imDepth;i++)
      qOverlayImage[i]->fill(Qt::transparent);

    overlayDataChanged();
}


void ImageViewerWidget::zoomIn()
{
    scale(1.25, false);
}

void ImageViewerWidget::zoomOut()
{
    scale(0.8, false);
}

void ImageViewerWidget::scale(double factor, bool absolute)
{
    if (absolute)
    {
        if (factor==scaleFactor)
          return;
        
        QGraphicsView::scale(factor/scaleFactor, factor/scaleFactor);
        scaleFactor = factor;
    }
    else
    {
        scaleFactor *= factor;
        QGraphicsView::scale(factor, factor);
    }

    displayHint(QString::number(int(scaleFactor*100)) + "%");
    emit(onRescaled(scaleFactor));
    emit(onScrollBarPositionChanged(horizontalScrollBar()->value(), verticalScrollBar()->value()));
}

void ImageViewerWidget::leaveEvent (QEvent * /*event*/)
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
    
    if (btn != Qt::LeftButton)
         return;
     
    if (cursorMode==cursorMove)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
    }
    
    QGraphicsView::mousePressEvent(event);
}

void ImageViewerWidget::mouseReleaseEvent ( QMouseEvent * event )
{
    Qt::MouseButton btn = event->button();
    
    if (line != 0 && cursorMode == cursorDrawLine) 
    {
    }
    else if (btn==Qt::LeftButton)
      setDragMode(QGraphicsView::NoDrag);
    
    QGraphicsView::mouseReleaseEvent(event);
}

void ImageViewerWidget::mouseMoveEvent(QGraphicsSceneMouseEvent* /*event*/)
{

}

void ImageViewerWidget::sceneMouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
    int x = int(event->scenePos().rx());
    int y = int(event->scenePos().ry());
    int z = slider->value();

    int w = qImage->width();
    int h = qImage->height();
    
    if (x>=0 && x<w && y>=0 && y<h)
    {
        if (cursorMode==cursorDraw || cursorMode==cursorDrawLine || cursorMode==cursorDrawBox)
          setCursor(Qt::CrossCursor);
        else setCursor(Qt::ArrowCursor);
        
        if (valueLblActivated)
        {
            valueLabel->show();
            displayPixelValue(x, y, z);
        }
        if (magnActivated)
        {
            displayMagnifyView(x, y, z);
            magnView->show();
        }
        
        if (cursorMode == cursorDrawLine)
        {
            setCursor(Qt::CrossCursor);
            if (event->buttons()==Qt::LeftButton)
            {
                QLineF newLine(line->line().p1(), event->scenePos());
                line->setLine(newLine);
                QString hint = "dx: " + QString::number(int(newLine.x2())-int(newLine.x1()));
                hint += " dy: " + QString::number(int(newLine.y2())-int(newLine.y1()));
                hint += "  len: " + QString::number(int(newLine.length()));
                displayHint(hint, 3000);
            }
        } 
        else if (cursorMode==cursorDraw && drawing)
        {
            QPainter painter(qOverlayImage[slider->sliderPosition()]);
            if (drawPen.color()==Qt::black)
                painter.setCompositionMode(QPainter::CompositionMode_Clear);
            painter.setPen(drawPen);
            painter.drawLine(event->scenePos(), QPoint(lastPixX, lastPixY));
            overlayDataChanged(false);
        }
        lastPixX = x;
        lastPixY = y;
        lastPixZ = z;
        
        
    }
    else
    {
        setCursor(Qt::ArrowCursor);
        valueLabel->hide();
        magnView->hide();
        lastPixX = -1;
        lastPixY = -1;
    }
    
}

void ImageViewerWidget::sceneMousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    int x = int(event->scenePos().rx());
    int y = int(event->scenePos().ry());

    int w = qImage->width();
    int h = qImage->height();
    
    if (x>=0 && x<w && y>=0 && y<h)
    {
        if(cursorMode==cursorDrawLine) 
        {
            if (imScene->items().contains(line))
              imScene->removeItem(line);
            QLineF newLine(event->scenePos(), event->scenePos());
            line->setLine(newLine);
            imScene->addItem(line);
        }
        else if (event->buttons()==Qt::LeftButton && cursorMode==cursorDraw)
        {
            drawing = true;
        }
    }
    lastPixX = x;
    lastPixY = y;
}

void ImageViewerWidget::sceneMouseReleaseEvent ( QGraphicsSceneMouseEvent * event )
{
    if (cursorMode==cursorDrawLine && imScene->items().contains(line))
      displayProfile(true);
    else if (cursorMode==cursorDraw && drawing)
    {
        QPainter painter(qOverlayImage[slider->sliderPosition()]);
        if (drawPen.color()==Qt::black)
            painter.setCompositionMode(QPainter::CompositionMode_Clear);
        painter.setPen(drawPen);
        painter.drawLine(event->scenePos(), QPoint(lastPixX, lastPixY));
        overlayDataChanged();
        drawing = false;
    }
}

void ImageViewerWidget::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    emit(onScrollBarPositionChanged(horizontalScrollBar()->value(), verticalScrollBar()->value()));
}

void ImageViewerWidget::setScrollBarPosition(int x, int y)
{
    if (horizontalScrollBar()->value()==x && verticalScrollBar()->value()==y)
      return;
    
    horizontalScrollBar()->setValue(x);
    verticalScrollBar()->setValue(y);
}

void ImageViewerWidget::wheelEvent ( QWheelEvent * event )
{
    if (event->modifiers() & Qt::ControlModifier)
    {
        if (event->delta()>0)
//           zoomInAct->trigger();
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
    case Qt::Key_R:
        autoRange = !autoRange; // Switch auto range mode
        if (autoRange)
          displayHint(QString("Autorange on"));
        else
          displayHint(QString("Autorange off"));
        redrawImage();
        break;
    case Qt::Key_H:
        displayHistogram();
        break;
    case Qt::Key_P:
        if (imScene->items().contains(line))
          displayProfile();
        break;
    }

    emit onKeyPressEvent(event);
}

void ImageViewerWidget::dragMoveEvent(QDragMoveEvent *de)
{
    de->accept();
}

void ImageViewerWidget::dragEnterEvent(QDragEnterEvent *event)
{
    if (event->mimeData()->hasUrls())
      event->acceptProposedAction();
}

void ImageViewerWidget::linkViewer(ImageViewerWidget* viewer)
{
    connect(this, SIGNAL(onRescaled(double)), viewer, SLOT(scale(double)));
    connect(this, SIGNAL(onScrollBarPositionChanged(int,int)), viewer, SLOT(setScrollBarPosition(int,int)));
    connect(slider, SIGNAL(valueChanged(int)), viewer->slider, SLOT(setValue(int)) );
    connect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), viewer, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));
    linkedWidgets.append(viewer);
    emit onRescaled(scaleFactor);
    emit(onScrollBarPositionChanged(horizontalScrollBar()->value(), verticalScrollBar()->value()));
    viewer->slider->setValue(slider->value());

}

void ImageViewerWidget::unlinkViewer(ImageViewerWidget* viewer)
{
    disconnect(this, SIGNAL(onRescaled(double)), viewer, SLOT(scale(double)));
    disconnect(this, SIGNAL(onScrollBarPositionChanged(int,int)), viewer, SLOT(setScrollBarPosition(int,int)));
    disconnect(slider, SIGNAL(valueChanged(int)), viewer->slider, SLOT(setValue(int)) );
    disconnect(imScene, SIGNAL(onMouseMove(QGraphicsSceneMouseEvent*)), viewer, SLOT(sceneMouseMoveEvent(QGraphicsSceneMouseEvent*)));
    linkedWidgets.removeAll(viewer);
}

void ImageViewerWidget::setCursorMode(const int &mode)
{
    cursorMode = mode;
    
    if (mode==cursorDraw || mode==cursorDrawLine || mode==cursorDrawBox)
        setCursor(Qt::CrossCursor);
    else setCursor(Qt::ArrowCursor);
    
    if (mode==cursorDraw)
    {
      colorPicker->show();
      if (qOverlayImage.isEmpty())
        createOverlayImage();
    }
    else
      colorPicker->hide();
}

void ImageViewerWidget::setDrawPenColor(const QColor &color)
{
    drawPen.setColor(color);
}

void ImageViewerWidget::showContextMenu(const QPoint& pos)
{
    QPoint globalPos = this->mapToGlobal(pos);

    QMenu contMenu;
    QAction *act;
    
    QMenu selectMenu("Tools");
    act = selectMenu.addAction("Hand"); act->setCheckable(true); act->setChecked(cursorMode==cursorMove);
    act = selectMenu.addAction("Draw"); act->setCheckable(true); act->setChecked(cursorMode==cursorDraw);
    act = selectMenu.addAction("Line"); act->setCheckable(true); act->setChecked(cursorMode==cursorDrawLine);
    act = selectMenu.addAction("Box"); act->setCheckable(true); act->setChecked(cursorMode==cursorDrawBox);
    contMenu.addMenu(&selectMenu);
    
    if (cursorMode==cursorDraw)
    {
        contMenu.addAction("Color...");
        contMenu.addAction("Width...");
        contMenu.addAction("Clear Overlay");
    }
    
    QMenu linkMenu("Link");
    int wIndex = 0;
    foreach(QWidget *widget, QApplication::topLevelWidgets()) 
    {
        if(widget!=this && widget->isWindow() && widget->metaObject()->className()==QString("ImageViewerWidget"))
        {
            act = linkMenu.addAction(widget->windowTitle());
            act->setData(wIndex);
            act->setCheckable(true);
            if (linkedWidgets.contains(static_cast<ImageViewerWidget*>(widget)))
            {
                QFont aFont = act->font();
                aFont.setBold(true);
                act->setFont(aFont);
                act->setChecked(true);
            }
        }
        wIndex++;
    }
    contMenu.addMenu(&linkMenu);
    contMenu.addAction(actionMap["help"]);
    contMenu.addAction(actionMap["saveAs"]);
    
    QAction* selectedItem = contMenu.exec(globalPos);
    if (selectedItem)
    {
        if (selectedItem->parentWidget()==&selectMenu)
        {
            if (selectedItem->text()=="Draw")
              setCursorMode(cursorDraw);
            else if (selectedItem->text()=="Line")
              setCursorMode(cursorDrawLine);
            else if (selectedItem->text()=="Box")
              setCursorMode(cursorDrawBox);
            else
              setCursorMode(cursorMove);
        }
        else if (selectedItem->parentWidget()==&linkMenu)
        {
            QWidget *widget = QApplication::topLevelWidgets()[selectedItem->data().toInt()];
            ImageViewerWidget *w = static_cast<ImageViewerWidget*>(widget);
            if(linkedWidgets.contains(w))
              unlinkViewer(w);
            else
              linkViewer(w);
        }
        else if (selectedItem->text()=="Color...")
        {
            colorPicker->popup();
        }
        else if (selectedItem->text()=="Width...")
        {
            bool ok;
            int lWidth = QInputDialog::getInt(this, tr(""), tr("Line width:"), drawPen.width(), 1, 10, 1, &ok);
            if (ok)
                drawPen.setWidth(lWidth);
        }        
        else if (selectedItem->text()=="Clear Overlay")
        {
            clearOverlay();
        }
    }
    else
    {
        // nothing was chosen
    }
}

void ImageViewerWidget::showHelp()
{
    smil::Gui::showHelp();
}


