/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef ImageViewerWidget_H
#define ImageViewerWidget_H

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>

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
    
    QStatusBar *statusBar;
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
