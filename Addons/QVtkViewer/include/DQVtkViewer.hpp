/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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

#include "QVtkViewerWidget.h"
#include "Gui/include/private/DImageViewer.hpp"
#include "Core/include/DTypes.h"
#include "Base/include/private/DMeasures.hpp"

namespace smil
{


    template <class T> class Image;
    
   /**
    * \ingroup Addons
    * \defgroup QVtkViewer
    * @{
    */
   

    /**
     * QVtk image viewer
     * 
     * Requires the Qt, Vtk and QVtk libraries.
     * 
     * Keyboard shortcuts:
     */
    template <class T>
    class QVtkViewer : public ImageViewer<T>, protected QVtkViewerWidget
    {
    public:
        typedef ImageViewer<T> parentClass;
        QVtkViewer()
          : ImageViewer<T>(), 
            QVtkViewerWidget()
        {
            if (is_same<T,UINT8>::value)
              imageImport->SetDataScalarTypeToUnsignedChar();
            else if (is_same<T,UINT16>::value)
              imageImport->SetDataScalarTypeToUnsignedShort();
            else if (is_same<T,INT16>::value)
              imageImport->SetDataScalarTypeToShort();
            
            setAutoRange(false);

            colorOpacityTransfertFunction->AddSegment(0, 0., 1, 0.);
            colorOpacityTransfertFunction->AddSegment(1, 1., ImDtTypes<T>::max(), 1.0);
            
            initLookup(ImDtTypes<T>::max());
        }
        QVtkViewer(Image<T> &im)
          : ImageViewer<T>(), 
          QVtkViewerWidget()
        {
            if (is_same<T,UINT8>::value)
              imageImport->SetDataScalarTypeToUnsignedChar();
            else if (is_same<T,UINT16>::value)
              imageImport->SetDataScalarTypeToUnsignedShort();
            else if (is_same<T,INT16>::value)
              imageImport->SetDataScalarTypeToShort();
            
            setImage(im);
            
            setAutoRange(false);
            
            colorOpacityTransfertFunction->AddSegment(0, 0., 1, 0.);
            colorOpacityTransfertFunction->AddSegment(1, 1., ImDtTypes<T>::max(), 1.0);
            
            initLookup(ImDtTypes<T>::max());
        }
        ~QVtkViewer()
        {
        }
        
        enum RepresentationType { NONE, COMPOSITE, MIP };
        
        virtual void setImage(Image<T> &im)
        {
            ImageViewer<T>::setImage(im);
            
            size_t imSize[3];
            this->image->getSize(imSize);
            imageImport->SetWholeExtent(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
            imageImport->SetDataExtent(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
            imageImport->SetImportVoidPointer(this->image->getVoidPointer());
            
            cube->SetBounds(0, imSize[0]-1, 0, imSize[1]-1, 0, imSize[2]-1);
            cube->Update();
            
            camera->SetFocalPoint(imSize[0]/2, imSize[1]/2, imSize[2]/2);
            int d = 2.5 * (imSize[0] > imSize[1] ? imSize[0] : imSize[1]);
            camera->SetPosition(imSize[0], imSize[1]/2, -d);
        }
        
        virtual void setAutoRange(bool on)
        {
            opacityTransfertFunction->RemoveAllPoints();
            if (on)
            {
                vector<T> r = rangeVal(*this->image);
                opacityTransfertFunction->AddSegment(r[0], 0., r[1], 1.0);
            }
            else
              opacityTransfertFunction->AddSegment(ImDtTypes<T>::min(), 0., ImDtTypes<T>::max(), 1.0);
        }
        
        virtual void onSizeChanged(size_t /*width*/, size_t /*height*/, size_t /*depth*/)
        {
        }
        
        virtual void drawImage()
        {
//             T rVals[2];
//             rangeVal(*this->image, rVals);
//             opacityTransfertFunction->RemoveAllPoints();
//             opacityTransfertFunction->AddSegment(rVals[0], 0., rVals[1], 1.0);
            
            QVtkViewerWidget::update();
        }
        
        virtual void hide()
        {
            QVtkViewerWidget::hide();
        }
        virtual void show()
        {
            QVtkViewerWidget::showNormal();
            this->drawImage();
        }
        virtual void show(Image<T> &im) 
        {
            this->setImage(im);
            this->show();
        }
        virtual void showLabel()
        {
            QVtkViewerWidget::showLabel();
            this->drawImage();
        }
        virtual bool isVisible()
        {
            return QVtkViewerWidget::isVisible();
        }
        virtual void setName(const char *_name)
        {
            QString buf = _name + QString(" (") + QString(parentClass::image->getTypeAsString()) + QString(")");
            QVtkViewerWidget::setWindowTitle(buf);
        }
        virtual void drawOverlay(const Image<T> &/*im*/)
        {
        }
        virtual void clearOverlay() {  }

        virtual void setCurSlice(int)
        {
        }
        
        
        void setRepresentationType(RepresentationType type)
        {
            QVtkViewerWidget::setRepresentationType(QVtkViewerWidget::RepresentationType(type));
        }
        
        void showAxes()
        {
             QVtkViewerWidget::showAxes();
        }

        void hideAxes()
        {
            QVtkViewerWidget::hideAxes();
        }

        void setInterpolationTypeToLinear()
        {
            QVtkViewerWidget::setInterpolationTypeToLinear();
        }
        
        void setInterpolationTypeToNearest()
        {
            QVtkViewerWidget::setInterpolationTypeToNearest();
        }
        
    protected:
        void initLookup(int typeMax)
        {
            QVtkViewerWidget::initLookup(typeMax);
        }
    };

    /*@}*/
    
} // namespace smil


#endif // _D_QVTK_IMAGE_VIEWER_HPP
