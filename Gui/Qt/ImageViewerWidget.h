#ifndef ImageViewerWidget_H
#define ImageViewerWidget_H

#include <QGraphicsScene>
#include <QGraphicsView>

#include "MagnifyView.h"



class QImageGraphicsScene : public QGraphicsScene
{
    Q_OBJECT
public:
    QImageGraphicsScene(QObject *parent=0);
    void mouseMoveEvent ( QGraphicsSceneMouseEvent * event );
signals:
    void onMouseMove(QGraphicsSceneMouseEvent* event);
};


class ImageViewerWidget : public QGraphicsView
{
    Q_OBJECT

public:
    ImageViewerWidget(QWidget *parent=0);
    ~ImageViewerWidget();

    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void wheelEvent( QWheelEvent* );
    virtual void keyPressEvent(QKeyEvent *);

    void setName(const char *name);
private:
    QImage *image;
    double scaleFactor;
    QImageGraphicsScene *imScene;
    QGraphicsPixmapItem *pixItem;

    MagnifyView *magnView;
    QLabel *valueLabel;

    bool magnActivated;
    bool valueLblActivated;
    
    void setImageSize(int w, int h);
    void createActions();
    void connectActions();
    
    void updateTitle();

    QAction *openAct;
    QAction *printAct;
    QAction *exitAct;
    QAction *zoomInAct;
    QAction *zoomOutAct;
    QAction *normalSizeAct;
    QAction *fitToWindowAct;
    QAction *aboutAct;
    QAction *aboutQtAct;
    
    QString name;

public slots:
    void load(const QString fileName);
    void loadFromData(const uchar *data, int w, int h);
    void zoomIn();
    void zoomOut();
    void scale(double factor);
    void update();

private slots:
    void sceneMouseMoveEvent ( QGraphicsSceneMouseEvent * event );

signals:
    void onCursorPixelValueChanged(int x, int y, int pixVal, bool insideImage);
    void onRescaled(double scaleFactor);
    void onDataChanged();
    void onKeyPressEvent(QKeyEvent *);
};




#endif // ImageViewerWidget_H
