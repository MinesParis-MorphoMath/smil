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

#include <QMenu>

#include <QColorDialog>
#include <QInputDialog>
#include <qwt_legend.h>

#include "PlotWidget.h"

PlotWidget::PlotWidget(QWidget *parent)
  : QwtPlot(parent)
{
    resize(480,280);
    setCanvasBackground(Qt::white);
    
    currentCurve = new QwtPlotCurve();
    currentCurve->attach(this);
    insertLegend(new QwtLegend(), QwtPlot::RightLegend);
    
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
}

void PlotWidget::saveCurrentCurve()
{
    currentCurve = new QwtPlotCurve();
    
#if QWT_VERSION < 0x060000
    QwtData *data = currentCurve->data().copy();
    currentCurve->setData(*data);
#else // QWT_VERSION < 0x060000
    QwtPointSeriesData *data = new QwtPointSeriesData(((QwtPointSeriesData*)(currentCurve->data()))->samples());
    currentCurve->setData(data);
#endif // QWT_VERSION < 0x060000

    currentCurve->attach(this);
    replot();
  
}

void PlotWidget::clearOtherCurves()
{
    for (QwtPlotItemList::const_iterator it=itemList().begin();it!=itemList().end();it++)
      if (*it!=currentCurve)
	(*it)->detach();
    replot();
}

void PlotWidget::removeCurve()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    curve->detach();
    replot();
}

void PlotWidget::showCurve()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    curve->show();
    replot();
}

void PlotWidget::hideCurve()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    curve->hide();
    replot();
}

void PlotWidget::chooseCurveColor()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    QColorDialog diag(this);
    QPen pen(curve->pen());
    QColor color = diag.getColor(pen.color(), this);
    if (color.isValid())
    {
	pen.setColor(color);
	curve->setPen(pen);
	replot();
    }
}

void PlotWidget::chooseCurveWidth()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    QPen pen(curve->pen());
    bool ok;
    int lWidth = QInputDialog::getInt(this, tr(""),
			  tr("Line width:"), pen.width(), 1, 10, 1, &ok);
    if (ok)
    {
	pen.setWidth(lWidth);
	curve->setPen(pen);
	replot();
    }
}

void PlotWidget::chooseCurveTitle()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    bool ok;
    QString title = QInputDialog::getText(this, tr(""),
				  tr("Curve Legend"), QLineEdit::Normal,
				  curve->title().text(), &ok);
    if (ok)
    {
	curve->setTitle(title);
	replot();
	update();
    }
}

void PlotWidget::chooseCurveBrushColor()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    QColorDialog diag(this);
    QBrush brush(curve->brush());
    QColor color = diag.getColor(brush.color(), this);
    if (color.isValid())
    {
	brush.setColor(color);
	curve->setBrush(color);
	replot();
    }
}

void PlotWidget::setCurveStyle()
{
    int itemIndex = (((QAction *)sender())->data()).toInt();
    QwtPlotCurve *curve = (QwtPlotCurve *) this->itemList()[itemIndex];
    QString style = ((QAction*)sender())->text();
    if (style=="Lines")
      curve->setStyle(QwtPlotCurve::Lines);
    else if (style=="Sticks")
      curve->setStyle(QwtPlotCurve::Sticks);
    else if (style=="Steps")
      curve->setStyle(QwtPlotCurve::Steps);
    else if (style=="Dots")
      curve->setStyle(QwtPlotCurve::Dots);
    replot();
}

void PlotWidget::showContextMenu(const QPoint& pos)
{
    QMenu contMenu;
    
    int itemCount = itemList().count();
    
    contMenu.addAction("Duplicate current curve");
    if(itemCount>1)
      contMenu.addAction("Delete other curves");
    
    int i=0;
    for (QwtPlotItemList::const_iterator it=itemList().begin();it!=itemList().end();it++,i++)
    {
	QwtPlotCurve *curve = (QwtPlotCurve *)(*it);
	QMenu *curveMenu = new QMenu("#" + QString::number(i) + " " + curve->title().text());
	QAction *action;
	action = curveMenu->addAction("Title...", this, SLOT(chooseCurveTitle()));
	action->setData(i);
	action = curveMenu->addAction("Color...", this, SLOT(chooseCurveColor()));
	action->setData(i);
	action = curveMenu->addAction("Width...", this, SLOT(chooseCurveWidth()));
	action->setData(i);
	action = curveMenu->addAction("Brush Color...", this, SLOT(chooseCurveBrushColor()));
	action->setData(i);

	QMenu *styleMenu = new QMenu("Style");
	action = styleMenu->addAction("Lines", this, SLOT(setCurveStyle()));
	action->setData(i);
	if (curve->style()==QwtPlotCurve::Lines) { action->setCheckable(true); action->setChecked(true); }
	action = styleMenu->addAction("Sticks", this, SLOT(setCurveStyle()));
	action->setData(i);
	if (curve->style()==QwtPlotCurve::Sticks) { action->setCheckable(true); action->setChecked(true); }
	action = styleMenu->addAction("Steps", this, SLOT(setCurveStyle()));
	action->setData(i);
	if (curve->style()==QwtPlotCurve::Steps) { action->setCheckable(true); action->setChecked(true); }
	action = styleMenu->addAction("Dots", this, SLOT(setCurveStyle()));
	action->setData(i);
	if (curve->style()==QwtPlotCurve::Dots) { action->setCheckable(true); action->setChecked(true); }
	curveMenu->addMenu(styleMenu);
	
	if (curve->isVisible())
	{
	    action = curveMenu->addAction("Hide", this, SLOT(hideCurve()));
	    action->setData(i);
	}
	else
	{
	    action = curveMenu->addAction("Show", this, SLOT(showCurve()));
	    action->setData(i);
	}
	if (i<itemCount-1)
	{
	    action = curveMenu->addAction("Remove", this, SLOT(removeCurve()));
	    action->setData(i);
	}
	contMenu.addMenu(curveMenu);
    }
    
    QPoint globalPos = this->mapToGlobal(pos);
    QAction* selectedItem = contMenu.exec(globalPos);
    if (selectedItem)
    {
	if (selectedItem->text()=="Duplicate current curve")
	{
	    saveCurrentCurve();
	}
	else if (selectedItem->text()=="Delete other curves")
	{
	    clearOtherCurves();
	}
    }
}
