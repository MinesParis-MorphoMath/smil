/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#ifndef _D_OPENCV_IMAGE_HPP
#define _D_OPENCV_IMAGE_HPP

#include <opencv/cv.h>

#include "Core/include/private/DImage.hxx"
#include "Core/include/private/DSharedImage.hpp"



namespace smil
{

   /**
    * \ingroup Addons
    * \defgroup OpenCVInterface OpenCV Interface
    * @{
    */
   
   /**
    * OpenCV Image Interface
    */
    template <class T>
    class OpenCVInt : public SharedImage<T>
    {
    public:
        typedef SharedImage<T> parentClass;
        
        //! Constructor
        OpenCVInt(cv::Mat &_cvMat)
        {
            BaseObject::className = "OpenCVInt";
            parentClass::init();
            this->pixels = (T*)(_cvMat.data);
            this->setSize(_cvMat.size().width, _cvMat.size().height);
        }
        //! Constructor
        OpenCVInt(IplImage *cvIm)
        {
            BaseObject::className = "OpenCVInt";
            parentClass::init();
            this->pixels = (T*)(cvIm->imageData);
            this->setSize(cvIm->width, cvIm->height);
        }
        
    #ifdef SWIGPYTHON
    private:
        struct python_iplimage
        {
            PyObject_HEAD
            IplImage *img;
            PyObject *data;
            size_t offset;
        };
        
    public:
        OpenCVInt(PyObject *obj)
        {
            python_iplimage *pIm = (python_iplimage*)obj;
            IplImage *cvIm = pIm->img;
            if (!cvIm)
            {
                cout << "Error: Input object must be an IplImage." << endl;
                return;
            }
            BaseObject::className = "OpenCVInt";
            parentClass::init();
            this->pixels = (T*)(cvIm->imageData);
            this->setSize(cvIm->width, cvIm->height);
        }
    #endif // SWIGPYTHON
    
    };
    
    template <class T>
    IplImage *toIplImage(Image<T> &im)
    {
    }
    
    template <class T>
    int getCvType()
    {
      return -1;
    }
    template<> int getCvType<UINT8>() { return CV_8UC1; }
    template<> int getCvType<UINT16>() { return CV_16UC1; }
    
    /**
     * Create a OpneCV Mat from a Smil image (the data is copied)
     */    
    template <class T>
    cv::Mat toMatImage(Image<T> &im)
    {
        int cvType = getCvType<T>();
        
        ASSERT(cvType!=-1, "Data type conversion not implemented", cv::Mat())
        
        return cv::Mat(im.getHeight(), im.getWidth(), cvType, im.getPixels());
    }



   /*@}*/
    
} // namespace smil

#endif // _D_OPENCV_IMAGE_HPP
