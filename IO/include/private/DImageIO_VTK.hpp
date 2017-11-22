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


#ifndef _D_IMAGE_IO_VTK_HPP
#define _D_IMAGE_IO_VTK_HPP


#include <fstream>
#include <cstdlib>
#include <string>
#include <cctype>

#include "Core/include/private/DTypes.hpp"
#include "Core/include/private/DImage.hpp"
#include "IO/include/private/DImageIO.hpp"

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
    
    // Big/Little endian swap
    template <class T>
    void endswap(T *objp)
    {
        unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
        std::reverse(memp, memp + sizeof(T));
    }
    
    RES_T readVTKHeader(ifstream &fp, VTKHeader &hStruct);
    RES_T getVTKFileInfo(const char* filename, ImageFileInfo &fInfo);
    
    template <class T=void>
    class VTKImageFileHandler : public ImageFileHandler<T>
    {
      public:
        VTKImageFileHandler()
          : ImageFileHandler<T>("BMP"),
            littleEndian(false),
            writeBinary(true)
        {
        }
        
        bool littleEndian;
        
        virtual RES_T getFileInfo(const char* filename, ImageFileInfo &fInfo)
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
            
            if ( (typeid(T)==typeid(unsigned char) && scalarType!=ImageFileInfo::SCALAR_TYPE_UINT8)
                || (typeid(T)==typeid(unsigned short) && scalarType!=ImageFileInfo::SCALAR_TYPE_UINT16)
                || (typeid(T)==typeid(short) && scalarType!=ImageFileInfo::SCALAR_TYPE_INT16))
            {
                cout << "Error: input file type is " << hStruct.scalarTypeStr << endl;
                fp.close();
                return RES_ERR_IO;
            }
            
            int width = hStruct.width, height = hStruct.height, depth = hStruct.depth;            
            image.setSize(width, height, depth);
            typename Image<T>::volType slices = image.getSlices();
            typename Image<T>::sliceType curSlice;
            typename Image<T>::lineType curLine;
            
            // Return to the begining of the data
            fp.seekg(hStruct.startPos);
            
            double scalarCoeff = double(ImDtTypes<T>::max()) / hStruct.scalarCoeff;
            
            if (!hStruct.binaryFile)
            {
                double val;
                for (int z=0;z<depth;z++)
                {
                    curSlice = slices[z];
                    for (int y=height-1;y>=0;y--)
                    {
                        curLine = curSlice[y];
                        for (int x=0;x<width;x++)
                        {
                            fp >> val;
                            curLine[x] = static_cast<T>(val*scalarCoeff);
                        }
                    }
                }
            }

            else
            {
                // In binary version, values are written as chars
                
                for (int z=0;z<depth;z++)
                {
                    curSlice = slices[z];
                    for (int y=height-1;y>=0;y--)
                    {
                        curLine = curSlice[y];
                        
                        if (littleEndian || sizeof(T)==1)
                            fp.read((char*)curLine, sizeof(T)*width);
                        else
                        {
                            T val;
                            for (int i=0;i<width;i++)
                            {
                              fp.read((char*)&val, sizeof(T));
                              endswap(&val);
                              curLine[i] = val;
                            }
                        }
                    }
                }
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

            size_t width = image.getWidth();
            size_t height = image.getHeight();
            size_t depth = image.getDepth();
            size_t pixNbr = width*height*depth;
            
            fp << "# vtk DataFile Version 3.0" << endl;
            fp << "vtk output" << endl;
            fp << "BINARY" << endl;
            fp << "DATASET STRUCTURED_POINTS" << endl;
            fp << "DIMENSIONS " << width << " " << height << " " << depth << endl;
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
              fp << "LOOKUP_TABLE default" << endl;
            else
            {
                cerr << "not implemented (todo..)" << endl;
            }
            
            typename Image<T>::volType slices = image.getSlices();
            typename Image<T>::sliceType curSlice;
            typename Image<T>::lineType curLine;
            
            if (writeBinary)
            {
              // todo : make this generic
//                 fp.write((char*)pixels, sizeof(T)*pixNbr);
              
                for (size_t z=0;z<depth;z++)
                {
                    curSlice = slices[z];
                    for (int y=height-1;y>=0;y--)
                    {
                        curLine = curSlice[y];
                        if (littleEndian || sizeof(T)==1)
                            fp.write((char*)curLine, sizeof(T)*width);
                        else
                        {
                            T val;
                            for (size_t i=0;i<width;i++)
                            {
                              val = curLine[i];
                              endswap(&val);
                              fp.write((char*)&val, sizeof(T));
                            }
                        }
                    }
                }
            }

            fp.close();
            
            return RES_OK;
        }
    };
    
    template <>
    inline RES_T VTKImageFileHandler<void>::read(const char *, Image<void> &)
    {
        return RES_ERR;
    }

    template <>
    inline RES_T VTKImageFileHandler<void>::write(const Image<void> &, const char *)
    {
        return RES_ERR;
    }
    
    template <>
    inline RES_T VTKImageFileHandler<RGB>::read(const char *, Image<RGB> &)
    {
        return RES_ERR_NOT_IMPLEMENTED;
    }
    template <>
    inline RES_T VTKImageFileHandler<RGB>::write(const Image<RGB> &, const char *)
    {
        return RES_ERR_NOT_IMPLEMENTED;
    }

/*@}*/

} // namespace smil


#endif // _D_IMAGE_IO_VTK_HPP
