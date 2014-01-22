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

#include "QVtkViewerWidget.h"

#include <QMenu>

#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkEventQtSlotConnect.h"

QVtkViewerWidget::QVtkViewerWidget(QWidget *parent) :
    QWidget(parent)
{
    resize(400, 330);
    horizontalLayout = new QHBoxLayout(this);
    qvtkWidget = new QVTKWidget(this);
    horizontalLayout->addWidget(qvtkWidget);
   
//     connect(qvtkWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
   
    
    // VTK Objects
    
    renderer = vtkRenderer::New();
    renderWindow = qvtkWidget->GetRenderWindow();
    renderWindow->AddRenderer(renderer);
    interactor = renderWindow->GetInteractor();
    camera = renderer->GetActiveCamera();
    
    imageImport = vtkImageImport::New();

    representationType = NONE;

    volumeRayCastFunction = NULL;
    volumeRayCastMapper = vtkVolumeRayCastMapper::New();
    volumeRayCastMapper->SetInput(imageImport->GetOutput());
    // 	    volumeRayCastMapper->SetSampleDistance(0.1);

    opacityTransfertFunction = vtkPiecewiseFunction::New();

    volumeProperty = vtkVolumeProperty::New();
    volumeProperty->SetColor(opacityTransfertFunction);
    volumeProperty->SetScalarOpacity(opacityTransfertFunction);
    volumeProperty->SetInterpolationTypeToLinear();


    volume = vtkVolume::New();
    volume->SetMapper(volumeRayCastMapper);
    volume->SetProperty(volumeProperty);
    setRepresentationType(COMPOSITE);
    renderer->AddViewProp(volume);

    cube = vtkCubeSource::New();
    outline = vtkOutlineFilter::New();
    outline->SetInput(cube->GetOutput());
    outlineMapper = vtkPolyDataMapper::New();
    outlineMapper->SetInput(outline->GetOutput());
    outlineActor = vtkActor::New();
    outlineActor->SetMapper(outlineMapper);
    renderer->AddViewProp(outlineActor);

    axesActor = vtkAxesActor::New();
    orientationMarker = vtkOrientationMarkerWidget::New();
    orientationMarker->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    orientationMarker->SetOrientationMarker( axesActor );
    orientationMarker->SetInteractor( renderWindow->GetInteractor() );
    orientationMarker->SetViewport( 0.0, 0.0, 0.4, 0.4 );
    orientationMarker->SetEnabled( 1 );
    orientationMarker->InteractiveOn();
	
    camera->SetViewUp(0, -1, 0);
	    
    vtkQtEventConnect = vtkEventQtSlotConnect::New();
    vtkQtEventConnect->Connect(qvtkWidget->GetRenderWindow()->GetInteractor(), vtkCommand::RightButtonPressEvent, 
			    this, SLOT(showContextMenu(vtkObject*, unsigned long, void*, void*, vtkCommand*)),
			    NULL, 1.0
			   );
}

QVtkViewerWidget::~QVtkViewerWidget()
{
    delete qvtkWidget;
    delete horizontalLayout;
    
    // VTK Objects
       
    renderer->Delete();
    volume->Delete();
    volumeRayCastMapper->Delete();
    volumeRayCastFunction->Delete();
    opacityTransfertFunction->Delete();
    volumeProperty->Delete();
    imageImport->Delete();
    outlineActor->Delete();
    outlineMapper->Delete();
    outline->Delete();
    cube->Delete();
    orientationMarker->Delete();
    axesActor->Delete();
    
    vtkQtEventConnect->Delete();
    
}

void QVtkViewerWidget::update()
{
    volume->Update();
    interactor->Render();
    QWidget::update();
}

void QVtkViewerWidget::setRepresentationType(RepresentationType type)
{
    if (type==representationType)
      return;
    representationType = type;
    if (volumeRayCastFunction)
      volumeRayCastFunction->Delete();
    switch (type)
    {
      case COMPOSITE:
	volumeRayCastFunction = vtkVolumeRayCastCompositeFunction::New();
	break;
      case MIP:
	volumeRayCastFunction = vtkVolumeRayCastMIPFunction::New();
	((vtkVolumeRayCastMIPFunction*)volumeRayCastFunction)->SetMaximizeMethodToOpacity();
	break;
    }
    volumeRayCastMapper->SetVolumeRayCastFunction(volumeRayCastFunction);
    
    if (QVtkViewerWidget::isVisible())
      update();
}

void QVtkViewerWidget::showAxes()
{
    orientationMarker->SetEnabled(1);
    interactor->Render(); 
}

void QVtkViewerWidget::hideAxes()
{
    orientationMarker->SetEnabled(0);
    interactor->Render(); 
}

void QVtkViewerWidget::setInterpolationTypeToLinear()
{
    volumeProperty->SetInterpolationTypeToLinear();
    interactor->Render(); 
}

void QVtkViewerWidget::setInterpolationTypeToNearest()
{
    volumeProperty->SetInterpolationTypeToNearest();
    interactor->Render(); 
}


void QVtkViewerWidget::showContextMenu(vtkObject*, unsigned long, void*, void*, vtkCommand *command)
{
    // consume event so the interactor style doesn't get it
    command->AbortFlagOn();
    int* sz = interactor->GetSize();
    int* position = interactor->GetEventPosition();
    QPoint pos = QPoint(position[0], sz[1]-position[1]);
    QPoint globalPos = this->mapToGlobal(pos);

    QAction *act;
    
    QMenu reprMenu("Representation type");
    act = reprMenu.addAction("Composite");
    act->setCheckable(true);
    act->setChecked(representationType==COMPOSITE);
    act = reprMenu.addAction("MIP");
    act->setCheckable(true);
    act->setChecked(representationType==MIP);

    QMenu interpMenu("Interpolation type");
    act = interpMenu.addAction("Linear");
    act->setCheckable(true);
    act->setChecked(volumeProperty->GetInterpolationType()==VTK_LINEAR_INTERPOLATION);
    act = interpMenu.addAction("Nearest");
    act->setCheckable(true);
    act->setChecked(volumeProperty->GetInterpolationType()==VTK_NEAREST_INTERPOLATION);

    QMenu contMenu(qvtkWidget);
    contMenu.addMenu(&reprMenu);
    contMenu.addMenu(&interpMenu);
    
    act = contMenu.addAction("Show axes");
    act->setCheckable(true);
    act->setChecked(orientationMarker->GetEnabled());
    
    QAction* selectedItem = contMenu.exec(globalPos);
    if (selectedItem)
    {
	if (selectedItem->text()=="Composite")
	  setRepresentationType(COMPOSITE);
	else if (selectedItem->text()=="MIP")
	  setRepresentationType(MIP);
	else if (selectedItem->text()=="Show axes")
	{
	    if (orientationMarker->GetEnabled())
	      hideAxes();
	    else
	      showAxes();
	}
	else if (selectedItem->text()=="Linear")
	  setInterpolationTypeToLinear();
	else if (selectedItem->text()=="Nearest")
	  setInterpolationTypeToNearest();
    }
}
