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

#ifndef _D_AA_IMAGE_VIEWER_HPP
#define _D_AA_IMAGE_VIEWER_HPP

#include "Gui/include/private/DImageViewer.hpp"
#include "Core/include/DTypes.h"
#include "Base/include/private/DImageTransform.hpp"

#include <aalib.h>

namespace smil
{
  template <class T>
  class Image;

  /**
   * \ingroup Gui
   */
  /*@{*/

  /**
   * AA (Ascii Art) image viewer
   *
   * Requires the <a href="http://aa-project.sourceforge.net/aalib/"
   * target="_blank">AA lib</a>. To use it, you must set the option USE_AALIB to
   * ON.
   */
  template <class T>
  class AaImageViewer : public ImageViewer<T>
  {
  public:
    typedef ImageViewer<T> parentClass;
    AaImageViewer();
    AaImageViewer(Image<T> *im);
    ~AaImageViewer();
    virtual void hide();
    virtual void show();
    virtual bool isVisible();
    virtual void setName(const char *_name);
    virtual void clearOverlay()
    {
    }

  protected:
    aa_context  *context;
    int          createContext();
    virtual void drawImage();
  };

  template <class T>
  AaImageViewer<T>::AaImageViewer()
  {
    context = NULL;
  }

  template <class T>
  AaImageViewer<T>::AaImageViewer(Image<T> *im) : ImageViewer<T>(im)
  {
    context = NULL;
  }

  template <class T>
  int AaImageViewer<T>::createContext()
  {
    context = aa_autoinit(&aa_defparams);
    if (context == NULL) {
      fprintf(stderr, "Cannot initialize AA-lib. Sorry.\n");
      return -1;
    }
    return 0;
  }

  template <class T>
  AaImageViewer<T>::~AaImageViewer()
  {
    hide();
  }

  template <class T>
  void AaImageViewer<T>::show()
  {
    drawImage();
  }

  template <class T>
  void AaImageViewer<T>::hide()
  {
    if (context)
      aa_close(context);
    context = NULL;
  }

  template <class T>
  void AaImageViewer<T>::drawImage()
  {
    if (!context)
      createContext();

    aa_resize(context);

    double imW = this->image->getWidth();
    double imH = this->image->getHeight();
    double imR = imW / imH;

    double scrW = aa_imgwidth(context);
    double scrH = aa_imgheight(context);
    double scrR = scrW / scrH;

    size_t w, h;
    // find dimensions to fit screen
    if (scrR > imR) {
      w = imW * scrH / imH;
      h = scrH;
    } else {
      w = scrW;
      h = imH * scrW / imW;
    }
    w *= aa_imgheight(context) / aa_scrheight(context);

    Image<T> tmpIm(w, h);
    resize(*this->image, w, h, tmpIm);

    unsigned char              *data   = aa_image(context);
    typename Image<T>::lineType pixels = tmpIm.getPixels();
    double                      coeff =
        double(numeric_limits<UINT8>::max()) / double(numeric_limits<T>::max());

    for (int j = 0; j < scrH; j++)
      for (int i = 0; i < scrW; i++, data++) {
        if (i < w && j < h)
          *data = (UINT8) (coeff * double(*pixels++));
        else
          *data = 0;
      }

    aa_render(context, &aa_defrenderparams, 0, 0, aa_scrwidth(context),
              aa_scrheight(context));
    aa_flush(context);
  }

  template <class T>
  bool AaImageViewer<T>::isVisible()
  {
    return context != NULL;
  }

  template <class T>
  void AaImageViewer<T>::setName(const char *_name)
  {
  }

  /*@{*/

} // namespace smil

#endif // _D_AA_IMAGE_VIEWER_HPP
