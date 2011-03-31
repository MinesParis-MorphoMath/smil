#ifndef _BASE_IMAGE_OPERATIONS_HPP
#define _BASE_IMAGE_OPERATIONS_HPP

#include "DImage.hpp"
#include "DMemory.hpp"

#include "DBaseLineOperations.hpp"


template <class T>
class imageFunctionBase
{
  public:
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    lineType *alignedBuffers;
    UINT32 bufferNumber;
    UINT32 bufferLength;
    UINT32 bufferSize;
    
    imageFunctionBase() : alignedBuffers(NULL) {};
    ~imageFunctionBase() 
    {
	deleteAlignedBuffers();
    };
    
    inline lineType *createAlignedBuffers(UINT8 nbr, UINT32 len);
    inline void deleteAlignedBuffers();
    inline void copyLineToBuffer(T *line, UINT32 bufIndex); 
    inline void copyBufferToLine(UINT32 bufIndex, T *line);  
};

template <class T, class lineFunction_T>
class unaryImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    unaryImageFunction() {}
    inline RES_T operator()(imageType &imIn, imageType &ImOut)
    {
	return this->_exec(imIn, ImOut);
    }
    inline RES_T operator()(imageType &ImOut, T &value)
    {
	return this->_exec(ImOut, value);
    }
    
    inline RES_T _exec(imageType &imIn, imageType &imOut);
    inline RES_T _exec(imageType &imOut, T &value);
    
  protected:	    
    lineFunction_T lineFunction;
};


template <class T, class lineFunction_T>
class binaryImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    binaryImageFunction() {}
    inline RES_T operator()(imageType &imIn1, imageType &imIn2, imageType &ImOut) { return this->_exec(imIn1, imIn2, ImOut); }
    inline RES_T operator()(imageType &imIn, T value, imageType &ImOut) { return this->_exec(imIn, value, ImOut); }
    
    inline RES_T _exec(imageType &imIn1, imageType &imIn2, imageType &imOut);
    inline RES_T _exec(imageType &imIn, imageType &imInOut);
    inline RES_T _exec(imageType &imIn, T value, imageType &imOut);
    
  protected:	    
    lineFunction_T lineFunction;
};






#include "DBaseImageOperations.hxx"

#endif
