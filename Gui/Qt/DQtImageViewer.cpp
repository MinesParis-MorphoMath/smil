/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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

#ifdef USE_QT


#include "Gui/Qt/DQtImageViewer.hpp"
#include "Gui/Qt/PureQt/ImageViewerWidget.h"
#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DMeasures.hpp"

namespace smil
{

    template <>
    void QtImageViewer<UINT8>::drawImage()
    {
        int sliceNbr = slider->value();
        Image<UINT8>::sliceType lines = this->image->getSlices()[sliceNbr];

        size_t w = this->image->getWidth();
        size_t h = this->image->getHeight();

        if (!parentClass::labelImage && autoRange)
        {
            vector<UINT8> rangeV = rangeVal<UINT8>(*this->image);
            double floor = rangeV[0];
            double coeff = 255. / double(rangeV[1]-rangeV[0]);
            UINT8 *destLine;
            
            for (size_t j=0;j<h;j++,lines++)
            {
                Image<UINT8>::lineType pixels = *lines;
                destLine = this->qImage->scanLine(j);
                for (size_t i=0;i<w;i++)
        //           pixels[i] = 0;
                    destLine[i] = (UINT8)(coeff * (double(pixels[i]) - floor));
            }
        }
        else
          for (size_t j=0;j<h;j++, lines++)
              memcpy(qImage->scanLine(j), *lines, sizeof(uchar) * w);
    }



    #ifdef SMIL_WRAP_BIN

    // #include "DBitArray.h"

    template <>
    void QtImageViewer<BIN>::drawImage()
    {
        Image<BIN>::lineType pixels = this->image->getPixels();
        size_t w = this->image->getWidth();
        size_t h = this->image->getHeight();

        this->setImageSize(w, h);

        const BIN *lIn;
        UINT8 *lOut, *lEnd;
        size_t bCount = (w-1)/BIN::SIZE + 1;

        for (int j=0;j<h;j++)
        {
            lIn = pixels + j*bCount;
            lOut = this->qImage->scanLine(j);
            lEnd = lOut + w;

            for (int b=0;b<bCount;b++,lIn++)
            {
              BIN_TYPE bVal = (*lIn).val;

              for (int i=0;i<BIN::SIZE;i++,lOut++)
              {
                if (lOut==lEnd)
                  break;
                *lOut = bVal & (1 << i) ? 255 : 0;
              }
            }
        }
    }

    #endif // SMIL_WRAP_BIN

} // namespace smil


#endif // USE_QT
