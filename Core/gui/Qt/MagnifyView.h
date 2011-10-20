#ifndef MagnifyView_H
#define MagnifyView_H

#include <QLabel>
#include <QMouseEvent>
#include <QAction>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsTextItem>
#include <QGraphicsScene>
#include <QGraphicsPathItem>

class MagnifyView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit MagnifyView(QWidget *parent = 0);

    ~MagnifyView();

private:
    QImage *fullImage;

    int gridSize;
    QGraphicsScene *scene;
    QGraphicsPixmapItem *pixItem;
    QGraphicsPathItem *pathItem;

    QList<QGraphicsTextItem*> *textItemList;

    double scaleFactor;

    void mouseMoveEvent(QMouseEvent* pEvent);
signals:

public slots:
    void zoomIn();
    void zoomOut();
    void scaleImage(double factor);
    void displayAt(int x, int y);
    void setGridSize(int s);

    void setImage(QImage *img);
};

#endif // MagnifyView_H
