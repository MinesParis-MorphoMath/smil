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

#ifdef USE_QT

#include "Gui/Qt/DQtImageViewer.hpp"
#include "Gui/Qt/PureQt/ImageViewerWidget.h"
#include "Core/include/private/DImage.hpp"

#include "DBitArray.h"

namespace smil
{
  template <>
  void QtImageViewer<Bit>::drawImage()
  {
    int                   sliceNbr = slider->value();
    Image<Bit>::sliceType lines    = this->image->getSlices()[sliceNbr];

    size_t w = this->image->getWidth();
    size_t h = this->image->getHeight();

    const BitArray::INT_TYPE *lIn;
    UINT8 *                   lOut, *lEnd;
    UINT                      bCount = BitArray::INT_SIZE(w);

    for (size_t j = 0; j < h; j++) {
      lIn  = lines[j].intArray;
      lOut = this->qImage->scanLine(j);
      lEnd = lOut + w;

      for (size_t b = 0; b < bCount; b++, lIn++) {
        BitArray::INT_TYPE bVal = (*lIn);

        for (size_t i = 0; i < BitArray::INT_TYPE_SIZE; i++, lOut++) {
          if (lOut == lEnd)
            break;
          *lOut = (bVal & (1L << i)) ? 255 : 0;
        }
      }
    }
  }

} // namespace smil

#endif // USE_QT
