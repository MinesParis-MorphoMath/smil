
class vtkVolumeProperty;/*
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


#ifndef _D_QVTK_IMAGE_VIEWER_HPP
#define _D_QVTK_IMAGE_VIEWER_HPP

#include <QApplication>

#include "QVtkViewer.h"
#include "Gui/include/private/DImageViewer.hpp"
#include "DTypes.h"

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

namespace smil
{


    template <class T> class Image;
    
   /**
    * \ingroup Gui
    */
    /*@{*/

    /**
     * QVtk image viewer
     * 
     * Requires the Qt, Vtk and QVtk libraries.
     * 
     * Keyboard shortcuts:
     */
    template <class T>
    class QVtkImageViewer : public ImageViewer<T>, protected QVtkViewer
    {
    public:
	typedef ImageViewer<T> parentClass;
	QVtkImageViewer()
	  : QVtkViewer(),
	    ImageViewer<T>()
	{
	    initialize();
	}
	QVtkImageViewer(Image<T> &im)
	  : QVtkViewer(),
	    ImageViewer<T>(im)
	{
	    initialize();
	    setImage(im);
	}
	~QVtkImageViewer()
	{
	    finalize();
	}
	
	virtual void setImage(Image<T> &im)
	{
	    ImageViewer<T>::setImage(im);
	    
	    size_t imSize[3];
	    im.getSize(imSize);
	    imageImport->SetImportVoidPointer(im.getVoidPointer());
	    imageImport->SetWholeExtent(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
	    imageImport->SetDataExtent(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
	    
	    cube->SetBounds(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
	    
	    vtkCamera *camera = getRenderer()->GetActiveCamera();
	    camera->SetFocalPoint(imSize[0]/2, imSize[1]/2, imSize[2]/2);
	    int d = 3 * (imSize[0] > imSize[1] ? imSize[0] : imSize[1]);
	    camera->SetPosition(imSize[0], imSize[1]/2, imSize[2]/2 + d);
	    camera->SetViewUp(-1, 0, 0);
	}
	
	virtual void hide()
	{
	    QVtkViewer::hide();
	}
	virtual void show()
	{
	    this->drawImage();
	    QVtkViewer::show();
	}
	virtual void showLabel()
	{
	}
	virtual bool isVisible()
	{
	    return QVtkViewer::isVisible();
	}
	virtual void setName(const char *_name)
	{
	    QString buf = _name + QString(" (") + QString(parentClass::image->getTypeAsString()) + QString(")");
	    QVtkViewer::setWindowTitle(buf);
	}
	virtual void drawImage()
	{
	    setName(this->image->getName());
	    
	    volume->Update();
	    getRenderWindow()->GetInteractor( )->Render(); 
	}
	virtual void drawOverlay(Image<T> &im)
	{
	}
	virtual void clearOverlay() {  }

	virtual void setCurSlice(int)
	{
	}
	
	enum RayCastType { RAYCAST_COMPOSITE = 1, RAYCAST_MIP };
	
	void SetRayCastType(UINT type)
	{
	    if (type==volumeRayCastType)
	      return;
	    if (volumeRayCastFunction)
	      volumeRayCastFunction->Delete();
	    switch (type)
	    {
	      case RAYCAST_COMPOSITE:
		volumeRayCastFunction = vtkVolumeRayCastCompositeFunction::New();
		break;
	      case RAYCAST_MIP:
		volumeRayCastFunction = vtkVolumeRayCastMIPFunction::New();
		((vtkVolumeRayCastMIPFunction*)volumeRayCastFunction)->SetMaximizeMethodToOpacity();
		break;
	    }
	    volumeRayCastMapper->SetVolumeRayCastFunction(volumeRayCastFunction);
	}

    protected:
	
	void initialize()
	{
	    imageImport = vtkImageImport::New();
	    imageImport->SetDataScalarType(VTK_UNSIGNED_CHAR);
	    volumeRayCastType = 0;
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
	    SetRayCastType(RAYCAST_COMPOSITE);
	    getRenderer()->AddViewProp(volume);
	    
	    opacityTransfertFunction->RemoveAllPoints();
	    opacityTransfertFunction->AddSegment(ImDtTypes<T>::min(), 0., ImDtTypes<T>::max(), 1.0);
	    
	    cube = vtkCubeSource::New();
	    outline = vtkOutlineFilter::New();
	    outline->SetInput(cube->GetOutput());
	    outlineMapper = vtkPolyDataMapper::New();
	    outlineMapper->SetInput(outline->GetOutput());
	    outlineActor = vtkActor::New();
	    outlineActor->SetMapper(outlineMapper);
	    getRenderer()->AddViewProp(outlineActor);
	}
	
	void finalize()
	{
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
	}
	
	vtkImageImport *imageImport;
	UINT volumeRayCastType;
	vtkVolumeRayCastFunction *volumeRayCastFunction;
	vtkVolumeRayCastMapper *volumeRayCastMapper;
	vtkVolume *volume;
	vtkVolumeProperty *volumeProperty;
	vtkPiecewiseFunction *opacityTransfertFunction;
	
	vtkCubeSource *cube;
	vtkOutlineFilter *outline;
	vtkPolyDataMapper *outlineMapper;
	vtkActor *outlineActor;

    };

    /*@{*/
    
} // namespace smil


#endif // _D_QVTK_IMAGE_VIEWER_HPP
