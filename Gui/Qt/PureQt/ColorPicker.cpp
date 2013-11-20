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

#include "ColorPicker.h"

#include <QStyle>
#include <QPainter>
#include <QPen>
#include <QStyle>
#include <QApplication>

ColorButton::ColorButton(QWidget* parent): QPushButton(parent)
{
//     setFlat(true);
    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    setAutoDefault(false);
    setCheckable(true);
    
    iconSize = style()->pixelMetric(QStyle::PM_SmallIconSize);
    setFixedSize(iconSize + 6, iconSize + 4);
    
    colorIndex = -1;
    
    connect(this, SIGNAL(toggled(bool)), SLOT(buttonPressed(bool)));
}

ColorButton::~ColorButton()
{

}

void ColorButton::setColor(const QColor& col, const int index)
{
    color = col;
    if (index>=0)
    {
      colorIndex = index;
      setText(QString::number(index));
    }
    repaint();
}

void ColorButton::paintEvent(QPaintEvent* e)
{
    QPixmap pix(iconSize + 6, iconSize + 4);
    pix.fill(palette().button().color());

    QPainter p(&pix);

    int w = pix.width();
    int h = pix.height();
//     p.drawRect(0, 0, w, h);
    p.setPen(QPen(Qt::gray));
    p.setBrush(color);
    p.drawRect(0, 0, w, h);
    setIcon(QIcon(pix));

    QPushButton::paintEvent(e);
}

void ColorButton::buttonPressed(bool toggled)
{
    if (toggled)
      emit colorChosen(this);
}



ColorPannel::ColorPannel(QWidget* parent)
  : QFrame(parent, Qt::Popup)
{
    setMouseTracking(true);
    
    grid = new QGridLayout();
    grid->setMargin(1);
    grid->setSpacing(0);    
    setLayout(grid);
    
    lastButtonSelected = NULL;
}

ColorPannel::~ColorPannel()
{
    clearGrid();
    delete grid;
}

void ColorPannel::clearGrid()
{
    for (int i=0;i<grid->count();i++)
    {
	QLayoutItem *item = grid->itemAt(0);
	grid->removeItem(item);
	delete item->widget();
    }
}

void ColorPannel::hideEvent(QHideEvent* e)
{
    QWidget::hideEvent(e);
    emit(onHide());
}

void ColorPannel::setColors(const QVector< QRgb >& cols)
{
    clearGrid();
    
    int colorCount = cols.count();
    int colCount = 16;
    int rowCount = colorCount / colCount;
    
    ColorButton but;
    setFixedSize(colCount*but.width(), rowCount*but.height());
    int index = 0;
    for (int j=0;j<rowCount;j++)
      for (int i=0;i<colCount;i++)
      {
	  if (index>=colorCount)
	    break;
	  ColorButton *button = new ColorButton(this);
	  button->setColor(cols[index], index);
	  button->setText("");
	  button->setToolTip(QString::number(index));
	  connect(button, SIGNAL(colorChosen(ColorButton*)), SLOT(colorButtonSelected(ColorButton*)));
	  grid->addWidget(button, j, i);
	  index++;
      }
}

void ColorPannel::colorButtonSelected(ColorButton* button)
{
    emit(colorSelected(button->getColor(), button->colorIndex));
    if (lastButtonSelected)
      lastButtonSelected->setChecked(false);
    hide();
    lastButtonSelected = button;
}




ColorPicker::ColorPicker(QWidget *parent)
    : ColorButton(parent)
{
    setFlat(false);
    
    pannel = new ColorPannel(this);
    connect(pannel, SIGNAL(colorSelected(QColor,int)), SLOT(colorSelected(QColor,int)));
    connect(pannel, SIGNAL(onHide()), SLOT(pannelClosed()));
    
    setFixedSize(55,30);
    
    connect(this, SIGNAL(toggled(bool)), SLOT(buttonPressed(bool)));
}

ColorPicker::~ColorPicker()
{
    delete pannel;
}

void ColorPicker::popup()
{
    pannel->move(mapToGlobal(QPoint(0, height())));
    pannel->show();
}

void ColorPicker::buttonPressed(bool toggled)
{
    if (!toggled)
        return;

    popup();
}

void ColorPicker::pannelClosed()
{
    setChecked(false);
}

void ColorPicker::setColors(const QVector< QRgb >& cols)
{
    if (cols.count()==0)
      return;
    pannel->setColors(cols);
    setColor(cols[1], 1);
}

void ColorPicker::colorSelected(const QColor &col, const int& index)
{
    setColor(col, index);
    emit(colorChanged(col));
}

QColor ColorPicker::getColor(const QPoint &pos)
{
}


