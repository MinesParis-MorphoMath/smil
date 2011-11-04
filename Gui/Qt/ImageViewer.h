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
