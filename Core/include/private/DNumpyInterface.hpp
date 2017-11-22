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


#ifndef _D_NUMPY_INTERFACE_HPP
#define _D_NUMPY_INTERFACE_HPP


#include "Core/include/DNumpy.h"
#include "Core/include/private/DImage.hxx"
#include "Core/include/private/DSharedImage.hpp"



namespace smil
{

   /**
    * \ingroup Core
    * \defgroup NumpyInterface Numpy Interface
    * @{
    */
   
   /**
    * Numpy Array Interface
    */
    template <class T>
    class NumpyInt : public SharedImage<T>
    {
    public:
        typedef SharedImage<T> parentClass;
        
        //! Constructor
        NumpyInt(PyObject *obj)
        {
            BaseObject::className = "NumpyInt";
            parentClass::init();
            
            PyArrayObject *arr = (PyArrayObject *)(obj);
            
            int dim = PyArray_NDIM(arr);
            npy_intp *dims = PyArray_DIMS(arr);
            
            T* data = (T*)PyArray_DATA(arr);
            
            PyArray_Descr *descr = PyArray_DESCR(arr);
            if (descr->type_num!=getNumpyType(*this))
            {
                ERR_MSG("Wrong data type");
                return;
            }

            if (dim==3)
              this->attach(data, dims[0], dims[1], dims[2]);
            else if (dim==2)
              this->attach(data, dims[0], dims[1]);
            else if (dim==1)
              this->attach(data, dims[0], 1);
            
        }
    };
    
   /*@}*/
    
} // namespace smil

#endif // _D_NUMPY_INTERFACE_HPP
