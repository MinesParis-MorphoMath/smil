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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _D_BASE_IMAGE_H
#define _D_BASE_IMAGE_H

#include "DBaseObject.h"
#include "DSignal.h"
#include "DSlot.h"
#include "DCommon.h"
#include "DErrors.h"

#include "Gui/include/private/DImageViewer.hpp"

namespace smil
{
  class BaseImageViewer;
  template <class T>
  class ImageViewer;

  /**
   * Base Image class
   */
  class BaseImage : public BaseObject
  {
    typedef BaseObject parentClass;

  public:
    BaseImage(const char *_className = "BaseImage")
        : BaseObject(_className), updatesEnabled(true), width(0), height(0),
          depth(0), pixelCount(0), lineCount(0), sliceCount(0),
          allocated(false), allocatedSize(0)
    {
      onModified = Signal(this);
      onShow     = Signal(this);
    }

    BaseImage(const BaseImage &rhs)
        : BaseObject(rhs), updatesEnabled(true), width(0), height(0), depth(0),
          pixelCount(0), lineCount(0), sliceCount(0), allocated(false),
          allocatedSize(0)
    {
      onModified = Signal(this);
      onShow     = Signal(this);
    }

    virtual ~BaseImage();

    // Forbid implicit assignment operator
    BaseImage &operator=(const BaseImage &rhs);

  public:
    virtual void init();
    //! Get image width
    inline size_t getWidth() const
    {
      return width;
    }
    //! Get image height
    inline size_t getHeight() const
    {
      return height;
    }
    //! Get image depth (Z)
    inline size_t getDepth() const
    {
      return depth;
    }

    //! Get memory size (bytes)
    virtual size_t getAllocatedSize() const
    {
      return allocatedSize;
    }

    //! Get dimension (2D or 3D)
    inline UINT getDimension() const
    {
      if (depth > 1)
        return 3;
      else if (height > 1)
        return 2;
      else
        return 1;
    }

    //! Set image size
    //! Set image size and allocate it if @b doAllocate is true
    virtual RES_T setSize(size_t w, size_t h, size_t d = 1,
                          bool doAllocate = true) = 0;

    //! Get image size
    inline void getSize(size_t *w, size_t *h, size_t *d) const
    {
      *w = this->width;
      *h = this->height;
      *d = this->depth;
    }

#ifndef SWIGPYTHON
    //! Get image size
    inline void getSize(int *w, int *h, int *d) const
    {
      *w = this->width;
      *h = this->height;
      *d = this->depth;
    }
#endif // SWIGPYTHON

    //! Get image size
    inline void getSize(size_t s[3]) const
    {
      s[0] = this->width;
      s[1] = this->height;
      s[2] = this->depth;
    }

    //! Get image size
    inline void getSize(off_t s[3]) const
    {
      s[0] = this->width;
      s[1] = this->height;
      s[2] = this->depth;
    }

    //! Get image size
    inline void getSize(int s[3]) const
    {
      s[0] = this->width;
      s[1] = this->height;
      s[2] = this->depth;
    }

    //! Get the number of pixels
    inline size_t getPixelCount() const
    {
      return this->pixelCount;
    }
    //! Get the number of lines
    inline size_t getLineCount() const
    {
      return this->lineCount;
    }
    //! Get the number of slices(for 3D images)
    inline size_t getSliceCount() const
    {
      return this->sliceCount;
    }

    //! Check if the image is allocated
    inline bool isAllocated() const
    {
      return this->allocated;
    }

    //! Get the void* data array
    virtual void *getVoidPointer() = 0;
    //! Trigger modified event
    virtual void modified() = 0;

    /**
     * areCoordsInImage() - checks if the triplet (x, y, z) in inside the image
     * bounds.
     *
     * @param[in] x,y,z : coords of a point
     */
    inline bool areCoordsInImage(const off_t x, const off_t y,
                                 const off_t z = 0) const
    {
      if (x < 0 || y < 0 || z < 0)
        return false;
      if (x >= off_t(this->width) || y >= off_t(this->height) ||
          z >= off_t(this->depth))
        return false;
      return true;
    }

    /**
     * areCoordsInImage() - checks if the triplet (x, y, z) in inside the image
     * bounds.
     *
     * @param[in] x,y,z : coords of a point
     *
     * @overload
     */
    inline bool areCoordsInImage(const size_t x, const size_t y,
                                 const size_t z = 0) const
    {
      if (x >= size_t(this->width) || y >= size_t(this->height) ||
          z >= size_t(this->depth))
        return false;
      return true;
    }

    /**
     * isPointInImage() - checks if a Point is in inside the image
     * bounds.
     *
     * @param[in] p : coords of a point
     *
     *
     */
    inline bool isPointInImage(const IntPoint &p) const
    {
      if (p.x < 0 || p.y < 0 || p.z < 0)
        return false;
      if (p.x >= int(width) || p.y >= int(height) || p.z >= int(depth))
        return false;
      return true;
    }

    /**
     * isOffsetInImage() - checks if a buffer offset in inside the image
     * bounds.
     *
     * @param[in] offset : offset of a point in the image buffer
     */
    inline bool isOffsetInImage(const off_t offset) const
    {
      if (offset < 0)
        return false;
      if (offset > off_t(getPixelCount()))
        return false;
      return true;
    }

    /**
     * isOffsetInImage() - checks if a buffer offset in inside the image
     * bounds.
     *
     * @param[in] offset : offset of a point in the image buffer
     */
    inline bool isOffsetInImage(const size_t offset) const
    {
      if (offset > getPixelCount())
        return false;
      return true;
    }

    //! Get an offset for given x,y(,z) coordinates
    inline size_t getOffsetFromCoords(size_t x, size_t y, size_t z = 0) const
    {
      size_t vmax = std::numeric_limits<size_t>::max();
      if (x >= this->width)
        return vmax;
      if (y >= this->height)
        return vmax;
      if (z >= this->depth)
        return vmax;
      return z * this->width * this->height + y * this->width + x;
    }

    //! Get an offset for given x,y(,z) coordinates
    inline size_t getOffsetFromPoint(IntPoint &p) const
    {
      size_t vmax = std::numeric_limits<size_t>::max();
      if (p.x < 0 || p.y < 0 || p.z < 0)
        return vmax;
      if (p.x >= int(width))
        return vmax;
      if (p.y >= int(height))
        return vmax;
      if (p.z >= int(depth))
        return vmax;
      return p.z * width * height + p.y * width + p.x;
    }

    //! Get x,y(,z) coordinates for a given offset
    inline void getCoordsFromOffset(size_t off, size_t &x, size_t &y,
                                    size_t &z) const
    {
      z = off / (this->width * this->height);
      y = (off % (this->width * this->height)) / this->width;
      x = off % this->width;
    }

    //! Get x,y(,z) coordinates for a given offset
    inline void getCoordsFromOffset(off_t off, off_t &x, off_t &y,
                                    off_t &z) const
    {
      z = off / (this->width * this->height);
      y = (off % (this->width * this->height)) / this->width;
      x = off % this->width;
    }

    //! Get x,y(,z) coordinates for a given offset
    inline std::vector<size_t> getCoordsFromOffset(size_t off) const
    {
      std::vector<size_t> coords(3);

      coords[2] = off / (this->width * this->height);
      coords[1] = (off % (this->width * this->height)) / this->width;
      coords[0] = off % this->width;
      return coords;
    }

    inline IntPoint getPointFromOffset(size_t off) const
    {
      IntPoint pt;

      pt.z = off / (this->width * this->height);
      pt.y = (off % (this->width * this->height)) / this->width;
      pt.x = off % this->width;
      return pt;
    }

    //! Get the description of the image
    virtual std::string getInfoString(const char * = "") const
    {
      return {};
    }
    //! Get the type of the image as a string ("UINT8",...)
    virtual const char *getTypeAsString() = 0;

    //! Check if the image (viewer) is visible
    virtual bool isVisible()
    {
      return false;
    }
    //! Show the image (viewer)
    virtual void show(const char * = NULL, bool = false);
    //! Show the image (viewer) as false colors
    virtual void showLabel(const char * = NULL);
    //! Hide the image (viewer)
    virtual void hide() = 0;

    //! Load from file
    virtual RES_T load(const char * /*fileName*/)
    {
      return RES_ERR_NOT_IMPLEMENTED;
    }
    //! Save to file
    virtual RES_T save(const char * /*fileName*/)
    {
      return RES_ERR_NOT_IMPLEMENTED;
    }

#ifndef SWIG
    //! Get the viewer associated to the image
    virtual BaseImageViewer *getViewer() = 0;
#endif // SWIG

    bool   updatesEnabled;
    Signal onModified;
    Signal onShow;

  protected:
    size_t dataTypeSize;

    size_t width;
    size_t height;
    size_t depth;

    size_t pixelCount;
    size_t lineCount;
    size_t sliceCount;

    bool allocated;

    size_t allocatedSize;
  };

  class ImageFreezer
  {
  public:
    ImageFreezer(BaseImage &im, bool updateOnDelete = true)
        : image(&im), update(updateOnDelete)
    {
      imState           = im.updatesEnabled;
      im.updatesEnabled = false;
    }
    ~ImageFreezer()
    {
      image->updatesEnabled = imState;
      if (update)
        image->modified();
    }

  protected:
    BaseImage *image;
    bool       imState;
    bool       update;
  };

#ifndef SWIG

  /**
   * Check if all images in a list have the same size.
   * The list of images must be finished by NULL.
   */
  inline bool haveSameSize(const BaseImage *im, ...)
  {
    va_list vargs;

    va_start(vargs, im);
    if (!im->isAllocated())
      return false;
    size_t w = im->getWidth();
    size_t h = im->getHeight();
    size_t d = im->getDepth();

    BaseImage *obj;
    while ((obj = va_arg(vargs, BaseImage *))) {
      if (!obj->isAllocated())
        return false;
      if (obj->getWidth() != w)
        return false;
      if (obj->getHeight() != h)
        return false;
      if (obj->getDepth() != d)
        return false;
    }
    va_end(vargs);
    return true;
  }

  /**
   * Set the same size to a list of images.
   * The size applied corresponds to the size of the first input image
   */
  inline bool setSameSize(const BaseImage *im, ...)
  {
    if (!im->isAllocated())
      return false;

    va_list vargs;

    va_start(vargs, im);
    size_t w = im->getWidth();
    size_t h = im->getHeight();
    size_t d = im->getDepth();

    BaseImage *obj;

    while ((obj = va_arg(vargs, BaseImage *))) {
      if (obj->getWidth() != w || obj->getHeight() != h || obj->getDepth() != d)
        if (obj->setSize(w, h, d) != RES_OK)
          return false;
    }
    return true;
  }

  /**
   * Check if all images in a list are allocated.
   * The list of images must be finished by NULL.
   */
  inline bool areAllocated(const BaseImage *im, ...)
  {
    va_list vargs;

    va_start(vargs, im);
    if (!im->isAllocated())
      return false;

    const BaseImage *obj;
    while ((obj = va_arg(vargs, const BaseImage *)))
      if (!obj->isAllocated())
        return false;
    va_end(vargs);
    return true;
  }

#define CHECK_ALLOCATED(...) (areAllocated(__VA_ARGS__, NULL))
#define ASSERT_ALLOCATED(...)                                                  \
  ASSERT(CHECK_ALLOCATED(__VA_ARGS__), RES_ERR_BAD_ALLOCATION)

#define CHECK_SAME_SIZE(...)                                                   \
  (Core::getInstance()->autoResizeImages ? setSameSize(__VA_ARGS__, NULL)      \
                                         : haveSameSize(__VA_ARGS__, NULL))
#define ASSERT_SAME_SIZE(...)                                                  \
  ASSERT(CHECK_SAME_SIZE(__VA_ARGS__), RES_ERR_BAD_SIZE)

#endif // SWIG

} // namespace smil

#endif // _DBASE_IMAGE_H
