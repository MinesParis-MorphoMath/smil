#ifndef _D_BASE_IMAGE_H
#define _D_BASE_IMAGE_H

#include "D_Types.h"
#include "DCommon.h"


class _SMIL baseImage
{
public:
    baseImage();
    inline UINT getWidth() const {
        return width;
    }
    inline UINT getHeight() const {
        return height;
    }
    inline UINT getDepth() const {
        return depth;
    }

    inline UINT getPixelCount() const {
        return pixelCount;
    }
    inline UINT getLineCount() const {
        return lineCount;
    }
    inline UINT getSliceCount() const {
        return sliceCount;
    }

    inline bool isAllocated() const {
        return allocated;
    }

    virtual void* getVoidPointer() = 0;
    virtual void modified() = 0;


protected:
    UINT dataTypeSize;

    UINT width;
    UINT height;
    UINT depth;

    UINT sliceCount;
    UINT lineCount;
    UINT pixelCount;

    bool allocated;

};



#endif // _DBASE_IMAGE_H

