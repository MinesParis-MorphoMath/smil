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


#ifndef _D_VTK_IO_HPP
#define _D_VTK_IO_HPP

#include "Core/include/DCore.h"
#include "Core/include/private/DSharedImage.hpp"

#include <vtkDICOMImageReader.h>
#include <vtkImageFlip.h>
#include <vtkSmartPointer.h>
#include <vtkImageReslice.h>


namespace smil
{
    /**
     * \ingroup Addons
     * \defgroup VtkInterface VTK Image Interface
     * @{
     */
    
    
    /**
     * 
    * Read DICOM
    */

    template <class T>
    RES_T readDICOM(const char *dirName, Image<T> &outIm, bool autoReslice=true)
    {
        vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
        reader->SetDirectoryName(dirName);
        reader->Update();
        
        if (reader->GetOutput()->GetScalarType()!=getVtkType<T>())
        {
            ERR_MSG("Wrong image type");
            cout << "vtkImageData type is " << reader->GetOutput()->GetScalarTypeAsString() << endl;
            return RES_ERR;
        }
        
        double spc = reader->GetDataSpacing()[0];
        
        vtkSmartPointer<vtkImageReslice> reslice = vtkSmartPointer<vtkImageReslice>::New();
        reslice->SetInputConnection(reader->GetOutputPort());
        reslice->SetOutputSpacing(spc, spc, spc);
        
        vtkImageData *imData;

        if (autoReslice)
        {
            reslice->Update();
            imData = reslice->GetOutput();
        }
        else
            imData = reader->GetOutput();
        
        VtkInt<T> sIm(imData);
        
        if (sIm.isAllocated())
        {
            outIm.setSize(sIm);
            copy(sIm, outIm);
            return RES_OK;
        }
        return RES_ERR;
        
    }
    
    template <>
    RES_T readDICOM<RGB>(const char */*dirName*/, Image<RGB> &/*outIm*/, bool /*autoReslice*/)
    {
        return RES_ERR_NOT_IMPLEMENTED;
    }
    
    /**@}*/
    
} // namespace smil

#endif // _D_VTK_IO_HPP
