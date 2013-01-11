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

#include "ImageViewerApp.h"
#include "ui_ImageViewerApp.h"



ImageViewerApp::ImageViewerApp(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::ImageViewerApp)
{
    ui->setupUi(this);

    graphicsView = new ImageViewerWidget();
    ui->gridLayout->addWidget(graphicsView);

    pixelPosLabel = new QLabel;
    pixelPosLabel->setFrameShadow(QFrame::Sunken);
    pixelPosLabel->setMinimumWidth(75);
    pixelValLabel = new QLabel;
    pixelValLabel->setMinimumWidth(50);
    scaleLabel = new QLabel;
    ui->statusBar->addPermanentWidget(pixelPosLabel);
    ui->statusBar->addPermanentWidget(pixelValLabel);
    ui->statusBar->addPermanentWidget(scaleLabel);

    connectActions();
}

ImageViewerApp::~ImageViewerApp()
{
    delete ui;
}

void ImageViewerApp::setName(const char *new_name)
{
    name = new_name;
    setWindowTitle(name);
}

void ImageViewerApp::load(const QString fileName)
{
    graphicsView->load(fileName);
}



void ImageViewerApp::viewMouseMoveEvent ( QMouseEvent * event )
{
    QPoint p = event->pos();

    pixelPosLabel->setText(QString("pos: " + QString::number(p.x()) + ", " + QString::number(p.y())));
}

void ImageViewerApp::displayPixelData(int x, int y, int pixVal, bool insideImage)
{
    if (insideImage)
    {
        pixelPosLabel->setText(QString("pos: " + QString::number(x) + ", " + QString::number(y)));
        pixelValLabel->setText(QString("val: " + QString::number(pixVal)));
    }
    else
    {
        pixelPosLabel->setText(QString(""));
        pixelValLabel->setText(QString(""));
    }
//    if (x>=0 && x<image->width() && y>=0 && y<image->height())
//    {
//        pixelPosLabel->setText(QString("pos: " + QString::number(x) + ", " + QString::number(y)));
//        pixelValLabel->setText(QString("val: " + QString::number(image->scanLine(y)[x])));
//    }
//    else
//    {
//        pixelPosLabel->setText(QString(""));
//        pixelValLabel->setText(QString(""));
//    }
}

void ImageViewerApp::displayScaleFactor(double sf)
{
    scaleLabel->setText(QString::number(int(sf*100.0)) + "%");
}

void ImageViewerApp::connectActions()
{
//     openAct = new QAction(tr("&Open..."), this);
//     openAct->setShortcut(tr("Ctrl+O"));
//     connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

//     printAct = new QAction(tr("&Print..."), this);
//     printAct->setShortcut(tr("Ctrl+P"));
//     printAct->setEnabled(false);
//     connect(printAct, SIGNAL(triggered()), this, SLOT(print()));

//     exitAct = new QAction(tr("E&xit"), this);
//     exitAct->setShortcut(tr("Ctrl+Q"));
//     connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

    connect(ui->zoomInAct, SIGNAL(triggered()), graphicsView, SLOT(zoomIn()));
    connect(ui->zoomOutAct, SIGNAL(triggered()), graphicsView, SLOT(zoomOut()));

    connect(graphicsView, SIGNAL(onCursorPixelValueChanged(int,int,int,bool)), this, SLOT(displayPixelData(int,int,int,bool)));


//      ui->menuView->addAction(ui->zoomInAct);
//      ui->menuView->addAction(ui->zoomOutAct);

//     normalSizeAct = new QAction(tr("&Normal Size"), this);
//     normalSizeAct->setShortcut(tr("Ctrl+S"));
//     normalSizeAct->setEnabled(false);
//     connect(normalSizeAct, SIGNAL(triggered()), this, SLOT(normalSize()));

//     fitToWindowAct = new QAction(tr("&Fit to Window"), this);
//     fitToWindowAct->setEnabled(false);
//     fitToWindowAct->setCheckable(true);
//     fitToWindowAct->setShortcut(tr("Ctrl+F"));
//     connect(fitToWindowAct, SIGNAL(triggered()), this, SLOT(fitToWindow()));

//     aboutAct = new QAction(tr("&About"), this);
//     connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

//     aboutQtAct = new QAction(tr("About &Qt"), this);
//     connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

