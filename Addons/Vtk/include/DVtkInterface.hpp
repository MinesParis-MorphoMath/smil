/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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


#ifndef _D_VTK_IMAGE_HPP
#define _D_VTK_IMAGE_HPP

#include "Core/include/DCore.h"
#include "Core/include/private/DSharedImage.hpp"

#include <vtkImageData.h>
#include <vtkImageFlip.h>

#if defined Py_PYCONFIG_H  || defined SWIGPYTHON
#include <vtkPythonUtil.h>
#endif

namespace smil
{
    /**
     * \ingroup Addons
     * \defgroup VtkInterface VTK Image Interface
     * @{
     */
    
    template <class T>
    int getVtkType()
    {
      return -1;
    }
    template<> int getVtkType<UINT8>() { return VTK_UNSIGNED_CHAR; }
    template<> int getVtkType<UINT16>() { return VTK_UNSIGNED_SHORT; }
    template<> int getVtkType<INT8>() { return VTK_SIGNED_CHAR; }
    template<> int getVtkType<INT16>() { return VTK_SHORT; }
    
    /**
     * 
    * VTK Image Interface
    */

    template <class T>
    class VtkInt : public SharedImage<T>
    {
    public:
        typedef SharedImage<T> parentClass;
        
        VtkInt(vtkImageData *imData, bool flipImage=true)
        {
            BaseObject::className = "VtkInt";
            parentClass::init();
            
            flip = vtkImageFlip::New();
            attach(imData, flipImage);
        }
    #if defined Py_PYCONFIG_H  || defined SWIGPYTHON
        VtkInt(PyObject *obj, bool flipImage=true)
        {
            BaseObject::className = "VtkInt";
            parentClass::init();
            
            flip = vtkImageFlip::New();
            
            vtkImageData * imData = (vtkImageData*)vtkPythonUtil::GetPointerFromObject(obj, "vtkImageData" );

            if ( imData == 0 ) // if the PyObject is not a vtk.vtkImageData object
            {
                PyErr_SetString( PyExc_TypeError, "Not a vtkImageData" );
            }
            else
            {
                attach(imData, flipImage);
            }
         }
    #endif // Py_PYCONFIG_H
    
          
          RES_T attach(vtkImageData *imData, bool flipImage)
          {
              if( getVtkType<T>()!=imData->GetScalarType() )
              {
                  ERR_MSG("Wrong image type");
                  cout << "vtkImageData type is " << imData->GetScalarTypeAsString() << endl;
                  return RES_ERR;
              }
              
              int *dims = imData->GetDimensions();

              if (flipImage)
              {
                  flip->SetInput(imData);
                  flip->SetFilteredAxis(1);
                  flip->Update();
                  
                  typename Image<T>::lineType pix = static_cast<typename Image<T>::lineType>(flip->GetOutput()->GetScalarPointer());
                  SharedImage<T>::attach(pix, dims[0], dims[1], dims[2]);
              }
              else
              {
                  typename Image<T>::lineType pix = static_cast<typename Image<T>::lineType>(imData->GetScalarPointer());
                  SharedImage<T>::attach(pix, dims[0], dims[1], dims[2]);
              }
              
              return RES_OK;
          }
    protected:
          vtkImageFlip *flip;
          bool _flipImage;
    };

    template <>
    class VtkInt<RGB>
    {
      public:
        VtkInt(vtkImageData *)
        {
        }
    };
    
    /**@}*/
    
} // namespace smil

#endif // _D_VTK_IMAGE_HPP
