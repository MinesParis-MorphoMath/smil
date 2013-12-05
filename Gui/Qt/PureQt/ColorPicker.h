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

#ifndef COLORPICKER_H
#define COLORPICKER_H

#include <QPushButton>
#include <QColor>
#include <QGridLayout>
#include <QEventLoop>
#include <QLabel>
#include <QEvent>
#include <QFocusEvent>

class ColorButton : public QPushButton
{
    Q_OBJECT
public:
    ColorButton(QWidget* parent = 0);
    ~ColorButton();

    QColor getColor()
    {
	return color;
    }
    int colorIndex;
public slots:
    virtual void setColor(const QColor &col, const int index=-1);
    
signals:
    void colorChosen(ColorButton *button);
  
protected:
    int iconSize;
    void paintEvent(QPaintEvent *e);

private slots:
    void buttonPressed(bool toggled);

private:
    QColor color;
};



class ColorPannel : public QFrame
{
    Q_OBJECT
    
public:
    ColorPannel(QWidget *parent=NULL);
    ~ColorPannel();
    
    void setColors(const QVector<QRgb> &cols);
    
signals:
    void colorSelected(const QColor &, const int &index);
    void onHide();
    
protected slots:
    void colorButtonSelected(ColorButton *button);
private:
    QGridLayout *grid;
protected:
    QVector<QRgb> colors;
    void hideEvent(QHideEvent *e);
    void clearGrid();
    ColorButton *lastButtonSelected;
};



class ColorPicker : public ColorButton
{
    Q_OBJECT


public:
    ColorPicker(QWidget *parent = 0);

    ~ColorPicker();

    QColor getColor(const QPoint &pos);
    void setColors(const QVector<QRgb> &cols);
    
    void popup();

protected slots:
    void colorSelected(const QColor &, const int &index);
    
    signals:
    void colorChanged(const QColor &);

protected:
    ColorPannel *pannel;

private slots:
    void buttonPressed(bool toggled);
    void pannelClosed();

private:
};

#endif
