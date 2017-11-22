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

#include "QVtkViewerWidget.h"

#include <QMenu>
#include <QKeyEvent>

#include <climits>

#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkEventQtSlotConnect.h"

QVtkViewerWidget::QVtkViewerWidget(QWidget *parent) :
    ImageViewerWidget(parent)
{
    resize(400, 330);
//     horizontalLayout = new QHBoxLayout(this);
    qvtkWidget = new QVTKWidget(this);
    layout->addWidget(qvtkWidget);
    this->hintLabel->setParent(qvtkWidget);
   
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
    volumeRayCastMapper->SetInputConnection(imageImport->GetOutputPort());
    //             volumeRayCastMapper->SetSampleDistance(0.1);

    opacityTransfertFunction = vtkPiecewiseFunction::New();
    colorOpacityTransfertFunction = vtkPiecewiseFunction::New();
    
    colorTransfertFunction = vtkDiscretizableColorTransferFunction::New();
    colorTransfertFunction->DiscretizeOn();
    
    volumeProperty = vtkVolumeProperty::New();
    volumeProperty->SetColor(opacityTransfertFunction);
    volumeProperty->SetScalarOpacity(opacityTransfertFunction);
    volumeProperty->SetInterpolationTypeToLinear();
//     volumeProperty->SetIndependentComponents(0);


    volume = vtkVolume::New();
    volume->SetMapper(volumeRayCastMapper);
    volume->SetProperty(volumeProperty);
    setRepresentationType(COMPOSITE);
    renderer->AddViewProp(volume);

    cube = vtkCubeSource::New();
    outline = vtkOutlineFilter::New();
    outline->SetInputConnection(cube->GetOutputPort());
    outlineMapper = vtkPolyDataMapper::New();
    outlineMapper->SetInputConnection(outline->GetOutputPort());
    outlineActor = vtkActor::New();
    outlineActor->SetMapper(outlineMapper);
    renderer->AddViewProp(outlineActor);

    axesActor = vtkAxesActor::New();
    orientationMarker = vtkOrientationMarkerWidget::New();
    orientationMarker->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    orientationMarker->SetOrientationMarker( axesActor );
    orientationMarker->SetInteractor( renderWindow->GetInteractor() );
    orientationMarker->SetViewport( 0.0, 0.0, 0.4, 0.4 );
    orientationMarker->SetEnabled( 0 );
        
    camera->SetViewUp(0, -1, 0);
            
    vtkQtEventConnect = vtkEventQtSlotConnect::New();
    vtkQtEventConnect->Connect(qvtkWidget->GetRenderWindow()->GetInteractor(), vtkCommand::RightButtonPressEvent, 
                            this, SLOT(showContextMenu(vtkObject*, unsigned long, void*, void*, vtkCommand*)),
                            NULL, 1.0
                           );
    vtkQtEventConnect->Connect(qvtkWidget->GetRenderWindow()->GetInteractor(), vtkCommand::KeyPressEvent, 
                            this, SLOT(keyPressed(vtkObject*, unsigned long, void*, void*, vtkCommand*)),
                            NULL, 1.0
                           );
}

QVtkViewerWidget::~QVtkViewerWidget()
{
    this->hintLabel->setParent(this->parentWidget());
    delete qvtkWidget;
//     delete horizontalLayout;
    
    // VTK Objects
       
    renderer->Delete();
    volume->Delete();
    volumeRayCastMapper->Delete();
    volumeRayCastFunction->Delete();
    opacityTransfertFunction->Delete();
    colorOpacityTransfertFunction->Delete();
    colorTransfertFunction->Delete();
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

void QVtkViewerWidget::initLookup(int typeMax)
{
    QVector<QColor> rainbowColorTable;
    for(int i=0;i<256;i++)
      rainbowColorTable.append(QColor::fromHsvF(double(i)/255., 1.0, 1.0));
    
    colorTransfertFunction->SetNumberOfValues(256);
    int factor = typeMax / 255;
    unsigned char curC = 0;
    for(int i=0;i<256;i++,curC+=47)
      colorTransfertFunction->AddRGBPoint(i*factor, rainbowColorTable[curC].redF(), rainbowColorTable[curC].greenF(), rainbowColorTable[curC].blueF());
    colorTransfertFunction->Build();
}

void QVtkViewerWidget::showNormal()
{
    volumeProperty->SetGradientOpacity((vtkPiecewiseFunction*)NULL);
    volumeProperty->SetColor(opacityTransfertFunction);
    volumeProperty->SetScalarOpacity(opacityTransfertFunction);
    volumeProperty->SetInterpolationTypeToLinear();
    show();
    interactor->Render();
}

void QVtkViewerWidget::showLabel()
{
    volumeProperty->SetGradientOpacity(colorOpacityTransfertFunction);
    volumeProperty->SetColor(colorTransfertFunction);
    volumeProperty->SetScalarOpacity(colorOpacityTransfertFunction);
    volumeProperty->SetInterpolationTypeToNearest();
    show();
    interactor->Render();
}

void QVtkViewerWidget::update()
{
    volume->Update();
    interactor->Render();
    QWidget::update();
}

void QVtkViewerWidget::setLabelImage(bool val)
{
    if (drawLabelized==val)
      return;
    
    drawLabelized = val;
    
    if (drawLabelized)
        showLabel();
    else
        showNormal();
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
      default:
        volumeRayCastFunction = vtkVolumeRayCastCompositeFunction::New();
        break;
    }
    volumeRayCastMapper->SetVolumeRayCastFunction(volumeRayCastFunction);
    
    if (QVtkViewerWidget::isVisible())
      update();
}

void QVtkViewerWidget::showAxes()
{
    orientationMarker->SetEnabled(1);
    orientationMarker->InteractiveOn();
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

void QVtkViewerWidget::keyPressed(vtkObject *caller, unsigned long, void*, void*, vtkCommand */*command*/)
{
    vtkRenderWindowInteractor *iren = static_cast<vtkRenderWindowInteractor*>(caller);
    char key = iren->GetKeyCode();
    
    switch(key)
    {
      case 'r':
        setAutoRange(!this->autoRange);
    }
    
    QKeyEvent kEvt(QEvent::KeyPress, key-'a' + Qt::Key_A, Qt::NoModifier);
    keyPressEvent(&kEvt);
}

