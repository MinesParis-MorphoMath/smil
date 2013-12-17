/*
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


#ifndef _D_IMAGE_IO_VTK_HPP
#define _D_IMAGE_IO_VTK_HPP


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <ctype.h>

#include "Core/include/private/DTypes.hpp"
#include "Core/include/private/DImage.hpp"
#include "IO/include/DCommonIO.h"

using namespace std;


namespace smil
{
    /** 
    * \addtogroup IO
    */
    /*@{*/
  
    struct VTKHeader
    {
	VTKHeader()
	  : width(0), height(0), depth(0),
	    pointNbr(0), scalarType(ImageFileInfo::SCALAR_TYPE_UNKNOWN),
	    scalarCoeff(0), binaryFile(false), startPos(0)
	{
	}
	
	UINT width, height, depth;
	UINT pointNbr;
	ImageFileInfo::ScalarType scalarType;
	string scalarTypeStr;
	double scalarCoeff;
	bool binaryFile;
	streampos startPos;
    };
    
    
    RES_T readVTKHeader(ifstream &fp, VTKHeader &hStruct);
    RES_T getVTKFileInfo(const char* filename, ImageFileInfo &fInfo);
    
    template <class T=void>
    class VTKImageFileHandler : public ImageFileHandler<T>
    {
      public:
	VTKImageFileHandler()
	  : ImageFileHandler<T>("BMP"),
	    writeBinary(true)
	{
	}
	
	virtual RES_T getFileInfo(const char* filename, ImageFileInfo &fInfo)
	{
	    std::ifstream fp;
	    ImageFileInfo::ScalarType scalarType = ImageFileInfo::SCALAR_TYPE_UINT8; // default, if not specified in the file header

	    /* open image file */
	    fp.open(filename, ios_base::binary);
	    
	//     fp.open(filename);
	    if (!fp)
	    {
		cerr << "error: couldn't open " << filename << "!" << endl;
		return RES_ERR;
	    }
	    
	    VTKHeader hStruct;
	    if (readVTKHeader(fp, hStruct)!=RES_OK)
	    {
		fp.close();
		ERR_MSG("Error reading VTK file header");
		return RES_ERR;
	    }
	    
	    fInfo.width = hStruct.width;
	    fInfo.height = hStruct.height;
	    fInfo.depth = hStruct.depth;
	    fInfo.colorType = ImageFileInfo::COLOR_TYPE_GRAY;
	    fInfo.scalarType = hStruct.scalarType;
	    
	    fp.close();
	    
	    return RES_OK;
	}
	
	bool writeBinary;
	
	virtual RES_T read(const char* filename, Image<T> &image)
	{
	    std::ifstream fp;

	    /* open image file */
	    fp.open(filename, ios_base::binary);
	    
	//     fp.open(filename);
	    if (!fp)
	    {
		cerr << "error: couldn't open " << filename << "!" << endl;
		return RES_ERR;
	    }

	    VTKHeader hStruct;
	    if (readVTKHeader(fp, hStruct)!=RES_OK)
	    {
		fp.close();
		ERR_MSG("Error reading VTK file header");
		return RES_ERR;
	    }
	    
	    ImageFileInfo::ScalarType scalarType = hStruct.scalarType==ImageFileInfo::SCALAR_TYPE_UNKNOWN ? ImageFileInfo::SCALAR_TYPE_UINT8 : hStruct.scalarType; // default, if not specified in the file header
	    
	    if ( (typeid(T)==typeid(unsigned char) && scalarType!=ImageFileInfo::SCALAR_TYPE_UINT8) ||
		 (typeid(T)==typeid(unsigned short) && scalarType!=ImageFileInfo::SCALAR_TYPE_UINT16))
	    {
		cout << "Error: input file type is " << hStruct.scalarTypeStr << endl;
		fp.close();
		return RES_ERR_IO;
	    }
	    	    
	    image.setSize(hStruct.width, hStruct.height, hStruct.depth);
	    typename Image<T>::lineType pixels = image.getPixels();
	    
	    // Return to the begining of the data
	    fp.seekg(hStruct.startPos);
	    
	    UINT ptsNbr = hStruct.pointNbr;
	    double scalarCoeff = double(ImDtTypes<T>::max()) / hStruct.scalarCoeff;
	    
	    if (!hStruct.binaryFile)
	    {
		double val;
		while(fp && --ptsNbr>0)
		{
		    fp >> val;
		    *pixels++ = (T)(val*scalarCoeff);
		}
		if(fp)
		{
		    fp >> val;
		    *pixels = (T)(val*scalarCoeff);
		}
	    }

	    else
	    {
		// In binary version, values are written as unsigned chars
		fp.read((char*)pixels, sizeof(T)*ptsNbr);
	    }

	    fp.close();

	    image.modified();
	    
	    return RES_OK;
	}
	virtual RES_T write(const Image<T> &image, const char* filename)
	{
	    std::ofstream fp;
	    
	    /* open image file */
	    fp.open(filename, ios_base::binary);
	    if (!fp)
	    {
		cerr << "error: couldn't open " << filename << "!" << endl;
		return RES_ERR;
	    }

	    size_t w = image.getWidth();
	    size_t h = image.getHeight();
	    size_t d = image.getDepth();
	    size_t pixNbr = w*h*d;
	    
	    fp << "# vtk DataFile Version 3.0" << endl;
	    fp << "vtk output" << endl;
	    fp << "BINARY" << endl;
	    fp << "DATASET STRUCTURED_POINTS" << endl;
	    fp << "DIMENSIONS " << w << " " << h << " " << d << endl;
	    fp << "SPACING 1.0 1.0 1.0" << endl;
	    fp << "ORIGIN 0 0 0" << endl;
	    fp << "POINT_DATA " << pixNbr << endl;
	    fp << "SCALARS scalars ";
	    if (typeid(T)==typeid(unsigned char))
	      fp << "unsigned_char";
	    else if (typeid(T)==typeid(unsigned short))
	      fp << "unsigned_short";
	    fp << endl;
	    
	    if (writeBinary)
	      fp << "COLOR_SCALARS ImageScalars 1" << endl;
	    else
	    {
		cerr << "not implemented (todo..)" << endl;
	    }
	    
	    typename Image<T>::lineType pixels = image.getPixels();
	    
	    if (writeBinary)
	    {
	      // todo : make this generic
		fp.write((char*)pixels, sizeof(T)*pixNbr);
	    }

	    fp.close();
	    
	    return RES_OK;
	}
    };
    
    template <>
    inline RES_T VTKImageFileHandler<void>::read(const char *filename, Image<void> &image)
    {
	return RES_ERR;
    }

    template <>
    inline RES_T VTKImageFileHandler<void>::write(const Image<void> &image, const char *filename)
    {
	return RES_ERR;
    }

/*@}*/

} // namespace smil


#endif // _D_IMAGE_IO_VTK_HPP
