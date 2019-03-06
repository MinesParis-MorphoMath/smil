/*
 * Smil
 * Copyright (c) 2011-2016 Matthieu Faessel
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

#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QWidget>
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
#include <vtkImageCast.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkEventQtSlotConnect.h>
#include <vtkDiscretizableColorTransferFunction.h>

#include "Gui/Qt/PureQt/ImageViewerWidget.h"

class QVtkViewerWidget : public ImageViewerWidget
{
  Q_OBJECT

public:
  typedef ImageViewerWidget parentClass;

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
  vtkPiecewiseFunction *colorOpacityTransfertFunction;
  vtkDiscretizableColorTransferFunction *colorTransfertFunction;

  vtkCubeSource *cube;
  vtkOutlineFilter *outline;
  vtkPolyDataMapper *outlineMapper;
  vtkActor *outlineActor;

  vtkAxesActor *axesActor;
  vtkOrientationMarkerWidget *orientationMarker;

  vtkEventQtSlotConnect *vtkQtEventConnect;

  void initLookup(int typeMax);

  void setRepresentationType(RepresentationType type);
  void showAxes();
  void showNormal();
  void showLabel();
  void hideAxes();
  void setInterpolationTypeToLinear();
  void setInterpolationTypeToNearest();

  //     virtual void keyPressEvent(QKeyEvent *event) {
  //     parentClass::keyPressEvent(event); }
  virtual void setLabelImage(bool val);
  virtual void setAutoRange(bool /*on*/){};
public slots:
  void showContextMenu(vtkObject *, unsigned long, void *, void *,
                       vtkCommand *command);
  void keyPressed(vtkObject *, unsigned long, void *, void *,
                  vtkCommand *command);
};

#endif
