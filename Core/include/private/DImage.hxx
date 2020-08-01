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


#ifndef _IMAGE_HXX
#define _IMAGE_HXX

#include <iostream>
#include <string>
#include <iomanip>

#include "Core/include/DCoreEvents.h"
#include "Base/include/private/DMeasures.hpp"
#include "Base/include/private/DImageArith.hpp"
#include "IO/include/private/DImageIO.hxx"
#include "Gui/include/DGuiInstance.h"

#ifdef SMIL_WRAP_RGB
#include "NSTypes/RGB/include/DRGB.h"
#endif // SMIL_WRAP_RGB

#ifdef SMIL_WRAP_BIT
#include "NSTypes/Bit/include/DBit.h"
#endif // SMIL_WRAP_BIT

namespace smil
{

    template <class T>
    Image<T>::Image()
      : BaseImage("Image")
    {
        init();
    }

    template <class T>
    Image<T>::Image(size_t w, size_t h, size_t d)
      : BaseImage("Image")
    {
        init();
        setSize(w, h, d);
    }

    template <class T>
    Image<T>::Image(const Image<T> &rhs, bool cloneData)
      : BaseImage(rhs)
    {
        init();
        if (cloneData)
          this->clone(rhs);
        else this->setSize(rhs);
    }

    template <class T>
    Image<T>::Image(const ResImage<T> &rhs, bool cloneData)
      : BaseImage(rhs)
    {
        init();
        if (cloneData)
          this->clone(rhs);
        else this->setSize(rhs);
    }

    template <class T>
    template <class T2>
    Image<T>::Image(const Image<T2> &rhs, bool cloneData)
      : BaseImage(rhs)
    {
        init();
        if (cloneData)
          this->clone(rhs);
        else setSize(rhs);
    }

    template <class T>
    Image<T>::Image(BaseImage *_im, bool stealIdentity)
      : BaseImage("Image")
    {
        init();

        if (_im==NULL)
          return;

        Image *im = castBaseImage(_im, T());
        if (im==NULL)
          return;

        if (!stealIdentity)
        {
            setSize(*im);
            return;
        }
        else
          drain(im, true);
    }


    template <class T>
    void Image<T>::drain(Image<T> *im, bool deleteSrc)
    {
        if (allocated)
          deallocate();

        this->width = im->width;
        this->height = im->height;
        this->depth = im->depth;

        this->sliceCount = im->sliceCount;
        this->lineCount = im->lineCount;
        this->pixelCount = im->pixelCount;

        this->pixels = im->pixels;
        this->slices = im->slices;
        this->lines = im->lines;

        this->allocated = im->allocated;
        this->allocatedSize = im->allocatedSize;

        this->name = im->name;
        this->viewer = im->viewer;

        im->allocated = false;

        if (deleteSrc)
          delete im;
    }


    template <class T>
    void Image<T>::clone(const Image<T> &rhs)
    {
        bool isAlloc = rhs.isAllocated();
        this->setSize(rhs, isAlloc);
        if (isAlloc)
          copyLine<T>(rhs.getPixels(), getPixelCount(), this->pixels);
        modified();
    }

    template <class T>
    template <class T2>
    void Image<T>::clone(const Image<T2> &rhs)
    {
        bool isAlloc = rhs.isAllocated();
        this->setSize(rhs, isAlloc);
        if (isAlloc)
          copy(rhs, *this);
        modified();
    }

    template <class T>
    Image<T>::Image(const char *fileName)
      : BaseImage("Image")
    {
        triggerEvents = true;
        init();
        read(fileName, *this);
    }

    template <class T>
    Image<T>::~Image()
    {
        if (viewer)
            delete viewer;
        viewer = NULL;

        this->deallocate();
    }



    template <class T>
    void Image<T>::init()
    {
        this->slices = NULL;
        this->lines = NULL;
        this->pixels = NULL;

        this->dataTypeSize = sizeof(pixelType);

        this->viewer = NULL;

        parentClass::init();
    }

    template <class T>
    RES_T Image<T>::load(const char *fileName)
    {
        return read(fileName, *this);
    }
    
    template <class T>
    RES_T Image<T>::save(const char *fileName)
    {
        return write(*this, fileName);
    }
    
    template <class T>
    void Image<T>::modified()
    {
        if (!this->updatesEnabled)
          return;

        BaseImageEvent event(this);
        onModified.trigger(&event);
    }



    template <class T>
    void Image<T>::setName(const char *_name)
    {
        parentClass::setName(_name);

        if (viewer)
            viewer->setName(_name);
    }

    template <class T>
    void Image<T>::createViewer()
    {
        if (viewer)
          return;

        viewer = Gui::getInstance()->createDefaultViewer<T>(*this);

    }

    template <class T>
    ImageViewer<T> *Image<T>::getViewer()
    {
        createViewer();
        return viewer;
    }

    template <class T>
    bool Image<T>::isVisible()
    {
        return (viewer && viewer->isVisible());
    }

    template <class T>
    void Image<T>::hide()
    {
      if (viewer)
        viewer->hide();
    }

    template <class T>
    void Image<T>::show(const char *_name, bool labelImage)
    {
        if (isVisible())
          return;

        if (!this->allocated)
        {
          ERR_MSG("Image isn't allocated !");
          return;
        }

        parentClass::show();
        
        createViewer();

        if (_name)
            setName(_name);

        if (!viewer)
          return;

        if (!labelImage)
          viewer->show();
        else
          viewer->showLabel();

    }

    template <class T>
    void Image<T>::showLabel(const char *_name)
    {
        show(_name, true);
    }

    template <class T>
    void Image<T>::showNormal(const char *_name)
    {
        if (_name)
            setName(_name);
        if (isVisible())
          viewer->show();
        else show(_name, false);
    }

    template <class T>
    RES_T Image<T>::setSize(size_t w, size_t h, size_t d, bool doAllocate)
    {
        if (w==this->width && h==this->height && d==this->depth)
            return RES_OK;

        if (this->allocated)
          this->deallocate();

        this->width = w;
        this->height = h;
        this->depth = d;

        this->sliceCount = d;
        this->lineCount = d * h;
        this->pixelCount = this->lineCount * w;

        if (doAllocate)
          ASSERT((this->allocate()==RES_OK));

        if (viewer)
          viewer->setImage(*this);

        this->modified();

        return RES_OK;
    }

    template <class T>
    RES_T Image<T>::allocate()
    {
        if (this->allocated)
            return RES_ERR_BAD_ALLOCATION;

        this->pixels = createAlignedBuffer<T>(pixelCount);
    //     pixels = new pixelType[pixelCount];

        ASSERT((this->pixels!=NULL), "Can't allocate image", RES_ERR_BAD_ALLOCATION);

        this->allocated = true;
        this->allocatedSize = this->pixelCount*sizeof(T);

        this->restruct();

        return RES_OK;
    }

    template <class T>
    RES_T Image<T>::restruct(void)
    {
        if (this->slices)
            delete[] this->slices;
        if (this->lines)
            delete[] this->lines;

        this->lines =  new lineType[lineCount];
        this->slices = new sliceType[sliceCount];

        sliceType cur_line = this->lines;
        volType cur_slice = this->slices;

        size_t pixelsPerSlice = this->width * this->height;

        for (size_t k=0; k<depth; k++, cur_slice++)
        {
          *cur_slice = cur_line;

          for (size_t j=0; j<height; j++, cur_line++)
            *cur_line = pixels + k*pixelsPerSlice + j*width;
        }


        return RES_OK;
    }


    template <class T>
    RES_T Image<T>::deallocate()
    {
        if (!this->allocated)
            return RES_OK;

        if (this->slices)
            delete[] this->slices;
        if (this->lines)
            delete[] this->lines;
        if (this->pixels)
            deleteAlignedBuffer<T>(pixels);

        this->slices = NULL;
        this->lines = NULL;
        this->pixels = NULL;

        this->allocated = false;
        this->allocatedSize = 0;

        return RES_OK;
    }

#ifndef SWIGPYTHON
    template <class T>
    void Image<T>::toArray(T outArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          outArray[i] = pixels[i];
    }

    template <class T>
    void Image<T>::fromArray(const T inArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          pixels[i] = inArray[i];
        modified();
    }

    template <class T>
    void Image<T>::toCharArray(signed char outArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          outArray[i] = pixels[i];
    }

    template <class T>
    void Image<T>::fromCharArray(const signed char inArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          pixels[i] = inArray[i];
        modified();
    }

    template <class T>
    void Image<T>::toIntArray(int outArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          outArray[i] = pixels[i];
    }

    template <class T>
    void Image<T>::fromIntArray(const int inArray[])
    {
        for (size_t i=0;i<pixelCount;i++)
          pixels[i] = inArray[i];
        modified();
    }
#endif // SWIGPYTHON

    template <class T>
    vector<int> Image<T>::toIntVector()
    {
        vector<int> vec;
        for (size_t i=0;i<pixelCount;i++)
          vec.push_back(pixels[i]);
        return vec;
    }

    template <class T>
    void Image<T>::fromIntVector(vector<int> inVector)
    {
        ASSERT((inVector.size()==pixelCount), "Vector length doesn't match image size.", );
        for (size_t i=0;i<min(pixelCount, inVector.size());i++)
          pixels[i] = inVector[i];
        modified();
    }

    template <class T>
    string Image<T>::toString()
    {
        string buf;
        for (size_t i=0;i<pixelCount;i++)
          buf.push_back(pixels[i]);
        return buf;
    }

    template <class T>
    void Image<T>::fromString(string pixVals)
    {
        ASSERT((pixVals.size()==pixelCount), "String length doesn't match image size.", );
        for (size_t i=0;i<pixelCount;i++)
          pixels[i] = pixVals[i];
        modified();
    }

    template <class T>
    void Image<T>::printSelf(ostream &os, bool displayPixVals, bool hexaGrid, string indent) const
    {
    #if DEBUG_LEVEL > 1
        cout << "Image::printSelf: " << this << endl;
    #endif // DEBUG_LEVEL > 1
        if (name!="")
          os << indent << "Image name: " << name << endl;

        if (depth>1)
          os << "3D image" << endl;
        else
          os << "2D image" << endl;

        T *dum = NULL;
        os << "Data type: " << getDataTypeAsString<T>(dum) << endl;

        if (depth>1)
          os << "Size: " << width << "x" << height << "x" << depth << endl;
        else
          os << "Size: " << width << "x" << height << endl;

        if (allocated) os << "Allocated (" << displayBytes(allocatedSize) << ")" << endl;
        else os << "Not allocated" << endl;


        if (displayPixVals)
        {
            std::stringstream tStr;
            tStr << (long)ImDtTypes<T>::max();
            size_t tSsize = tStr.str().size();
            if (hexaGrid)
              tSsize = size_t(tSsize * 1.5);


            os << "Pixel values:" << endl;
            size_t i, j, k;

            for (k=0;k<depth;k++)
            {
              for (j=0;j<height;j++)
              {
                if (hexaGrid && j%2)
                  os << setw(tSsize/2+1) << " ";
                for (i=0;i<width;i++)
                  os <<  setw(tSsize+1) << ImDtTypes<T>::toString(getPixel(i,j,k)) << ",";
                os << endl;
              }
              os << endl;
            }
            os << endl;
        }

        os << endl;
    }


    template <class T>
    SharedImage<T> Image<T>::getSlice(size_t sliceNum) const
    {
        if (sliceNum>=this->depth)
        {
          ERR_MSG("sliceNum > image depth");
          return SharedImage<T>();
        }
        return SharedImage<T>(*this->getSlices()[sliceNum], this->width, this->height, 1);
    }

    // OPERATORS

    template <class T>
    void operator << (ostream &os, const Image<T> &im)
    {
        im.printSelf(os);
    }

    template <class T>
    Image<T>& Image<T>::operator << (const char *s)
    {
        read(s, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator >> (const char *s)
    {
        write(*this, s);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator << (const Image<T> &rhs)
    {
        copy(rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator << (const T &value)
    {
        fill(*this, value);
        return *this;
    }

    template <class T>
    ResImage<T>Image<T>::operator ~() const
    {
        ResImage<T>im(*this);
        inv(*this, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator -() const
    {
        ResImage<T>im(*this);
        inv(*this, im);
        return im;
    }

    template <class T>
    ResImage<T> Image<T>::operator + (const Image<T> &rhs)
    {
        ResImage<T> im(*this);
        add(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T> Image<T>::operator + (const T &value)
    {
        ResImage<T> im(*this);
        add(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator += (const Image<T> &rhs)
    {
        add(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator += (const T &value)
    {
        add(*this, value, *this);
        return *this;
    }

    template <class T>
    ResImage<T>Image<T>::operator - (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        sub(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator - (const T &value)
    {
        ResImage<T>im(*this);
        sub(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator -= (const Image<T> &rhs)
    {
        sub(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator -= (const T &value)
    {
        sub(*this, value, *this);
        return *this;
    }

    template <class T>
    ResImage<T>Image<T>::operator * (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        mul(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator * (const T &value)
    {
        ResImage<T>im(*this);
        mul(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator *= (const Image<T> &rhs)
    {
        mul(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator *= (const T &value)
    {
        mul(*this, value, *this);
        return *this;
    }

    template <class T>
    ResImage<T>Image<T>::operator / (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        div(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator / (const T &value)
    {
        ResImage<T>im(*this);
        div(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator /= (const Image<T> &rhs)
    {
        div(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator /= (const T &value)
    {
        div(*this, value, *this);
        return *this;
    }

    template <class T>
    ResImage<T>Image<T>::operator == (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        equ(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator != (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        diff(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator < (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        low(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator < (const T &value)
    {
        ResImage<T>im(*this);
        low(*this, value, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator <= (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        lowOrEqu(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator <= (const T &value)
    {
        ResImage<T>im(*this);
        lowOrEqu(*this, value, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator > (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        grt(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator > (const T &value)
    {
        ResImage<T>im(*this);
        grt(*this, value, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator >= (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        grtOrEqu(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator >= (const T &value)
    {
        ResImage<T>im(*this);
        grtOrEqu(*this, value, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator | (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        sup(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator | (const T &value)
    {
        ResImage<T>im(*this);
        sup(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator |= (const Image<T> &rhs)
    {
        sup(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator |= (const T &value)
    {
        sup(*this, value, *this);
        return *this;
    }

    /** @cond */
    template <class T>
    ResImage<T>Image<T>::operator & (const Image<T> &rhs)
    {
        ResImage<T>im(*this);
        inf(*this, rhs, im);
        return im;
    }

    template <class T>
    ResImage<T>Image<T>::operator & (const T &value)
    {
        ResImage<T>im(*this);
        inf(*this, value, im);
        return im;
    }

    template <class T>
    Image<T>& Image<T>::operator &= (const Image<T> &rhs)
    {
        inf(*this, rhs, *this);
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator &= (const T &value)
    {
        inf(*this, value, *this);
        return *this;
    }
    /** @endcond */
    
    template <class T>
    Image<T>::operator bool()
    {
        return vol(*this)==ImDtTypes<T>::max()*pixelCount;
    }

    template <class T>
    Image<T>& Image<T>::operator << (const lineType &tab)
    {
        for (size_t i=0;i<pixelCount;i++)
          pixels[i] = tab[i];
        modified();
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator << (vector<T> &vect)
    {
        typename vector<T>::iterator it = vect.begin();
        typename vector<T>::iterator it_end = vect.end();

        for (size_t i=0;i<pixelCount;i++, it++)
        {
          if (it==it_end)
            break;
          pixels[i] = *it;
        }
        modified();
        return *this;
    }

    template <class T>
    Image<T>& Image<T>::operator >> (vector<T> &vect)
    {
        vect.clear();
        for (size_t i=0;i<pixelCount;i++)
        {
            vect.push_back(pixels[i]);
        }
        return *this;
    }

    typedef Image<UINT8> Image_UINT8;
    typedef Image<UINT16> Image_UINT16;
    typedef Image<UINT32> Image_UINT32;
    typedef Image<bool> Image_bool;
    

} // namespace smil

#if defined SWIGPYTHON && defined USE_NUMPY && defined(SWIG_WRAP_CORE)
#include "Core/include/DNumpy.h"

namespace smil {

    template <class T>
    PyObject * Image<T>::getNumArray(bool c_contigous)
    {
        npy_intp d[3];
        int dim = this->getDimension();
        if (dim==3)
        {
            d[0] = this->getDepth();
            d[1] = this->getHeight();
            d[2] = this->getWidth();
        }
        else
        {
            d[0] = this->getHeight();
            d[1] = this->getWidth();
        }
        PyObject *array = PyArray_SimpleNewFromData(dim, d, getNumpyType(*this), this->getPixels());

        if (c_contigous)
        {
            return array;
        }
        else
        {
            npy_intp t2[] = { 1, 0 };
            npy_intp t3[] = { 2, 1, 0 };
            PyArray_Dims trans_dims;
            if (dim==3)
              trans_dims.ptr = t3;
            else
              trans_dims.ptr = t2;
            trans_dims.len = dim;

            PyObject *res = PyArray_Transpose((PyArrayObject*) array, &trans_dims);
            Py_DECREF(array);
            return res;
        }
    }
    
    template <class T>
    void Image<T>::fromNumArray(PyObject *obj)
    {
        PyArrayObject *arr = NULL;
        PyArray_Descr *descr = NULL;

        if (PyArray_GetArrayParamsFromObject(obj, NULL, 1, &descr, NULL, NULL, &arr, NULL) != 0) 
        {
            ERR_MSG("Input must be a NumPy array");
            return;
        }
        descr = PyArray_DESCR(arr);
        if (descr && descr->type_num!=getNumpyType(*this))
        {
            ERR_MSG("Wrong input NumPy array data type");
            return;
        }
        int ndim = PyArray_NDIM(arr);
        if (ndim > 3) {
            ERR_MSG("Numpy array has more than three axes");
            return;        
        }
        npy_intp *dims = PyArray_DIMS(arr);
        for (int i = 0; i < 3; i++) {
          if (i >= ndim)
            dims[i] = 1;
        }
       
        setSize(dims[0], dims[1], dims[2]);
#if 0
        T *data = (T*)PyArray_DATA(arr);
        if (data)
        {
            for (size_t i=0;i<pixelCount;i++)
              pixels[i] = data[i];
        }
#else

#if 0
        for (npy_intp i = 0; i < dims[0]; i++) {
          for (npy_intp j = 0; j < dims[1]; j++) {
            for (npy_intp k = 0; k < dims[2]; k++) {
              T *v = (T *) PyArray_GETPTR3(arr, i, j, k);
              setPixel(i, j, k, *v);
            }
          }
        }
#else
        npy_intp ind[3];
        for (ind[0] = 0; ind[0] < dims[0]; ind[0]++) {
          for (ind[1] = 0; ind[1] < dims[1]; ind[1]++) {
            for (ind[2] = 0; ind[2] < dims[2]; ind[2]++) {
              T *v = (T *) PyArray_GetPtr(arr, ind);
              setPixel(ind[0], ind[1], ind[2], *v);
            }
          }
        }
#endif
#endif
        modified();
    }

} // namespace smil


#endif // defined SWIGPYTHON && defined USE_NUMPY

#endif // _IMAGE_HXX
