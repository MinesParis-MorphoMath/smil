/*
 * Smil
 * Copyright (c) 2011 Matthieu Faessel
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

#ifndef QVTK_VIEWER_H
#define QVTK_VIEWER_H

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>
#include "QVTKWidget.h"

class QVtkViewer : public QWidget
{
    Q_OBJECT
    
public:
    QVtkViewer(QWidget *parent = 0);
    ~QVtkViewer();
    
    QHBoxLayout *horizontalLayout;
    QVTKWidget *qvtkWidget;
    
    vtkRenderWindow *getRenderWindow();
    vtkRenderer *getRenderer();
    
protected:
    vtkRenderer *renderer;
};

#endif
