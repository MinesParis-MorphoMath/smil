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


#ifndef ImageViewerWidget_H
#define ImageViewerWidget_H

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QSlider>
#include <QGridLayout>


#include "MagnifyView.h"
#include "ColorPicker.h"
#include "Core/include/DBinary.h"
#include "Core/include/DCommon.h"

class QwtPointSeriesData;

class QImageGraphicsScene : public QGraphicsScene
{
    Q_OBJECT
public:
    QImageGraphicsScene(QObject *parent=0);
    void mouseMoveEvent (QGraphicsSceneMouseEvent *event);
    void mousePressEvent (QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent (QGraphicsSceneMouseEvent *event);
signals:
    void onMouseMove(QGraphicsSceneMouseEvent* event);
    void onMousePress(QGraphicsSceneMouseEvent* event);
    void onMouseRelease(QGraphicsSceneMouseEvent* event);
};


class ImageViewerWidget : public QGraphicsView
{
    Q_OBJECT

public:
    ImageViewerWidget(QWidget *parent=0);
    ~ImageViewerWidget();

    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void wheelEvent( QWheelEvent* );
    virtual void keyPressEvent(QKeyEvent *);
    virtual void leaveEvent (QEvent *event);
    
    virtual void setLabelImage(bool val);
    virtual void displayPixelValue(size_t, size_t, size_t) {}
    virtual void displayMagnifyView(size_t, size_t, size_t) {}
    virtual void displayMagnifyView() { displayMagnifyView(lastPixX, lastPixY, lastPixZ); }
    virtual void setCurSlice(int) {}
    virtual void redrawImage() {}
    virtual void createOverlayImage();
    virtual void deleteOverlayImage();

    void setName(QString name);
    void setImageSize(int w, int h, int d=1);
    void dataChanged();
    virtual void clearOverlay();
    
    QStatusBar *statusBar;
    QImage *qImage;
    QVector<QImage*> qOverlayImage;
    QImage::Format imageFormat;
    
    bool drawLabelized;
    // Auto adjust range
    bool autoRange;
    
    virtual void displayHistogram(bool = false) {}
    virtual void displayProfile(bool = false) {}
    
    void linkViewer(ImageViewerWidget *viewer);
    void unlinkViewer(ImageViewerWidget *viewer);
protected:
    QGridLayout *layout;
    
    QVector<QRgb> baseColorTable;
    QVector<QRgb> rainbowColorTable;
    QVector<QRgb> labelColorTable;
    QVector<QRgb> overlayColorTable;
    void initColorTables();
    void updatePixmaps(QImage *image, QList<QGraphicsPixmapItem*> *pixmaps);
    
    double scaleFactor;
    QImageGraphicsScene *imScene;
    QList<QGraphicsPixmapItem*> imagePixmaps;
    QList<QGraphicsPixmapItem*> overlayPixmaps;
    
    QLabel *valueLabel;
    QLabel *hintLabel;
    QTimer *hintTimer;
    QTimer *iconTimer;
    MagnifyView *magnView;

    size_t imWidth, imHeight, imDepth;
    int lastPixX, lastPixY, lastPixZ;
    
    bool magnActivated;
    bool valueLblActivated;
    
    void createActions();
    
    void updateTitle();
    void displayHint(QString msg, int timerVal=1000);

    QMap<QString, QAction*> actionMap;
    
    QString name;
    
    QSlider *slider;

    virtual void dropEvent(QDropEvent *) {}
    void dragMoveEvent(QDragMoveEvent *de);
    void dragEnterEvent(QDragEnterEvent *event);
    
    enum cursorMode { cursorMove, cursorDraw, cursorDrawLine, cursorDrawBox};
    int cursorMode;
    QGraphicsLineItem *line;
    
    QList<ImageViewerWidget*> linkedWidgets;
    
    ColorPicker *colorPicker;
    QPen drawPen;
    bool drawing;
    
    void scrollContentsBy(int dx, int dy);
public slots:
    void load(const QString fileName);
    void zoomIn();
    void zoomOut();
    void scale(double factor, bool absolute=true);
    void sliderChanged(int newVal)
    {
        displayHint(QString::number(newVal) + "/" + QString::number(slider->maximum()));
        setCurSlice(newVal);
        if (!qOverlayImage.isEmpty())
          updatePixmaps(qOverlayImage[newVal], &overlayPixmaps);
    }
    virtual void overlayDataChanged(bool triggerEvents=true);
    void updateIcon();
    void showContextMenu(const QPoint& pos);
    void mouseMoveEvent ( QGraphicsSceneMouseEvent * event );
    void setCursorMode(const int &mode);
    void setDrawPenColor(const QColor &color);
    void showHelp();
    void saveAs(const char *fileName=NULL);
    
protected slots:
    void setScrollBarPosition(int x, int y);
private slots:
    void sceneMousePressEvent ( QGraphicsSceneMouseEvent * event );
    void sceneMouseMoveEvent ( QGraphicsSceneMouseEvent * event );
    void sceneMouseReleaseEvent ( QGraphicsSceneMouseEvent * event );

signals:
    void onRescaled(double scaleFactor);
    void onDataChanged();
    void onKeyPressEvent(QKeyEvent *);
    void onScrollBarPositionChanged(int dx, int dy);
};




#endif // ImageViewerWidget_H
