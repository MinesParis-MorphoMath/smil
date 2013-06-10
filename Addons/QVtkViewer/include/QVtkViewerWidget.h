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


#include <vtkImageImport.h>
#include <vtkCamera.h>
#include <vtkVolumeRayCastCompositeFunction.h>
#include <vtkVolumeRayCastMIPFunction.h>
#include <vtkVolumeRayCastMapper.h>
#include <vtkVolume.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPiecewiseFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkCubeSource.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkEventQtSlotConnect.h>


class QVtkViewerWidget : public QWidget
{
    Q_OBJECT
    
public:
    QHBoxLayout *horizontalLayout;
    QVTKWidget *qvtkWidget;
    
    QVtkViewerWidget(QWidget *parent = 0);
    ~QVtkViewerWidget();
    
    
    enum RepresentationType { NONE, COMPOSITE, MIP };
    virtual void update();
    
protected:
    vtkRenderer *renderer;
    vtkRenderWindow *renderWindow;
    vtkRenderWindowInteractor *interactor;
    vtkCamera *camera;

    vtkImageImport *imageImport;
    RepresentationType representationType;
    vtkVolumeRayCastFunction *volumeRayCastFunction;
    vtkVolumeRayCastMapper *volumeRayCastMapper;
    vtkVolume *volume;
    vtkVolumeProperty *volumeProperty;
    vtkPiecewiseFunction *opacityTransfertFunction;
    
    vtkCubeSource *cube;
    vtkOutlineFilter *outline;
    vtkPolyDataMapper *outlineMapper;
    vtkActor *outlineActor;
    
    vtkAxesActor *axesActor;
    vtkOrientationMarkerWidget *orientationMarker;
    
    vtkEventQtSlotConnect *vtkQtEventConnect;
    
    
    void setRepresentationType(RepresentationType type);
    void showAxes();
    void hideAxes();
    void setInterpolationTypeToLinear();
    void setInterpolationTypeToNearest();
    void setAutorangeOn();
    void setAutorangeOff();
    
public slots:
    void showContextMenu(vtkObject*, unsigned long, void*, void*, vtkCommand *command);
    
};

#endif
