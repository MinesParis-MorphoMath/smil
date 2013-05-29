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

#include "QVtkViewer.h"

#include "vtkRenderer.h"
#include "vtkRenderWindow.h"

QVtkViewer::QVtkViewer(QWidget *parent) :
    QWidget(parent)
{
       resize(400, 330);
       horizontalLayout = new QHBoxLayout(this);
       qvtkWidget = new QVTKWidget(this);

       horizontalLayout->addWidget(qvtkWidget);
       
       renderer = vtkRenderer::New();
       qvtkWidget->GetRenderWindow()->AddRenderer(renderer);
}

QVtkViewer::~QVtkViewer()
{
       renderer = vtkRenderer::New();
       qvtkWidget->GetRenderWindow()->RemoveRenderer(renderer);
       renderer->Delete();
       delete qvtkWidget;
       delete horizontalLayout;
}


vtkRenderWindow *QVtkViewer::getRenderWindow()
{
    return qvtkWidget->GetRenderWindow();
}

vtkRenderer *QVtkViewer::getRenderer()
{
    return this->renderer;
}
