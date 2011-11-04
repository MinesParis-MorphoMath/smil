#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>

#include "ImageViewer.h"
#include "ui_ImageViewer.h"



ImageViewer::ImageViewer(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ImageViewer)
{
    ui->setupUi(this);

    graphicsView = new ImageViewerWidget();
    ui->gridLayout->addWidget(graphicsView);
    QGraphicsView *w = graphicsView;



//    QGraphicsScene *scn = new QGraphicsScene( w );
//    scn->setSceneRect( w->rect() );

//    w->setFixedSize( 400, 400 );


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

ImageViewer::~ImageViewer()
{
    delete ui;
}

void ImageViewer::setName(const char *new_name)
{
    name = new_name;
    setWindowTitle(name);
}

void ImageViewer::load(const QString fileName)
{
    graphicsView->load(fileName);
}

void ImageViewer::loadFromData(const uchar *data, int w, int h)
{
    graphicsView->loadFromData(data, w, h);
}



void ImageViewer::viewMouseMoveEvent ( QMouseEvent * event )
{
    QPoint p = event->pos();

    pixelPosLabel->setText(QString("pos: " + QString::number(p.x()) + ", " + QString::number(p.y())));
}

void ImageViewer::displayPixelData(int x, int y, int pixVal, bool insideImage)
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

void ImageViewer::displayScaleFactor(double sf)
{
    scaleLabel->setText(QString::number(int(sf*100.0)) + "%");
}

void ImageViewer::connectActions()
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

