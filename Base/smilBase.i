// Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




%include smilCommon.i
%include <std_container.i>

SMIL_MODULE(smilBase)


//////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////

%{
/* Includes the header in the wrapper code */
#include "Core/include/private/DImage.hxx"
#include "DImageArith.hpp"
#include "DImageDraw.h"
#include "DImageDraw.hpp"
#include "DImageHistogram.hpp"
#include "DImageTransform.hpp"
#include "DImageConvolution.hpp"
#include "DBaseMeasureOperations.hpp"
#include "DMeasures.hpp"
#include "DImageMatrix.hpp"
#include "DBlobMeasures.hpp"
#include "DBlobOperations.hpp"
#include "DImageCompare.hpp"

#include <stdexcept>

using namespace std;

%}

// Import smilCore to have correct function signatures (arguments with Image_UINT8 instead of Image<unsigned char>)
%import "smilCore.i"

PTR_ARG_OUT_APPLY(ret_min)
PTR_ARG_OUT_APPLY(ret_max)
PTR_ARG_OUT_APPLY(w)
PTR_ARG_OUT_APPLY(h)
PTR_ARG_OUT_APPLY(d)

%include "DImageArith.hpp"
%include "DImageDraw.h"
%include "DImageDraw.hpp"
%include "DImageTransform.hpp"


TEMPLATE_WRAP_FUNC_2T_CROSS(copy);
TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(copy);
TEMPLATE_WRAP_FUNC_2T_CROSS(cast);
TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(cast);

TEMPLATE_WRAP_FUNC(crop);
TEMPLATE_WRAP_SUPPL_FUNC(crop);
TEMPLATE_WRAP_FUNC(clone);

#ifdef SMIL_WRAP_RGB
%template(copyChannel) smil::copyChannel<RGB, UINT8>;
%template(copyToChannel) smil::copyToChannel<UINT8, RGB>;
%template(splitChannels) smil::splitChannels<RGB, UINT8>;
%template(mergeChannels) smil::mergeChannels<UINT8, RGB>;
#endif // SMIL_WRAP_RGB

TEMPLATE_WRAP_FUNC(inv);
TEMPLATE_WRAP_FUNC(fill);
TEMPLATE_WRAP_FUNC(randFill);
TEMPLATE_WRAP_FUNC(add);
TEMPLATE_WRAP_FUNC(addNoSat);
TEMPLATE_WRAP_FUNC(sub);
TEMPLATE_WRAP_FUNC(subNoSat);
TEMPLATE_WRAP_FUNC(mul);
TEMPLATE_WRAP_FUNC(mulNoSat);
TEMPLATE_WRAP_FUNC(div);
TEMPLATE_WRAP_FUNC_2T_CROSS(log);
TEMPLATE_WRAP_FUNC_2T_CROSS(exp);
TEMPLATE_WRAP_FUNC_2T_CROSS(pow);
TEMPLATE_WRAP_FUNC_2T_CROSS(sqrt);

TEMPLATE_WRAP_FUNC(equ);
TEMPLATE_WRAP_FUNC(diff);
TEMPLATE_WRAP_FUNC(absDiff);
TEMPLATE_WRAP_FUNC(sup);
TEMPLATE_WRAP_FUNC(inf);
TEMPLATE_WRAP_FUNC(low);
TEMPLATE_WRAP_FUNC(lowOrEqu);
TEMPLATE_WRAP_FUNC(grt);
TEMPLATE_WRAP_FUNC(grtOrEqu);
TEMPLATE_WRAP_FUNC(logicAnd);
TEMPLATE_WRAP_FUNC(logicOr);
TEMPLATE_WRAP_FUNC(logicXOr);
TEMPLATE_WRAP_FUNC(bitAnd);
TEMPLATE_WRAP_FUNC(bitOr);
TEMPLATE_WRAP_FUNC(bitXOr);
TEMPLATE_WRAP_FUNC_2T_CROSS(test);
TEMPLATE_WRAP_FUNC_2T_CROSS(compare);
TEMPLATE_WRAP_FUNC(mask);
TEMPLATE_WRAP_FUNC_2T_CROSS(applyLookup);

// Suppl. Types
TEMPLATE_WRAP_SUPPL_FUNC(fill);
TEMPLATE_WRAP_SUPPL_FUNC(equ);
TEMPLATE_WRAP_SUPPL_FUNC(diff);
TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(test);
TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(compare);

// *******************************
// Filters from Jose-Marcio
// *******************************
TEMPLATE_WRAP_FUNC_2T_CROSS(scaleRange);
// TEMPLATE_WRAP_FUNC_2T_CROSS(scaleRangeAuto);
TEMPLATE_WRAP_FUNC_2T_CROSS(sCurve);


%include "DImageHistogram.hpp"
TEMPLATE_WRAP_FUNC(histogram);

TEMPLATE_WRAP_FUNC_2T_CROSS(threshold);

TEMPLATE_WRAP_FUNC(histogramRange);
TEMPLATE_WRAP_FUNC_2T_CROSS(stretchHistogram);
TEMPLATE_WRAP_SUPPL_FUNC_2T_CROSS(stretchHistogram);
TEMPLATE_WRAP_FUNC(enhanceContrast);

TEMPLATE_WRAP_FUNC(otsuThresholdValues);
TEMPLATE_WRAP_FUNC_2T_CROSS(otsuThreshold);


%include "DImageConvolution.hpp"
TEMPLATE_WRAP_FUNC(horizConvolve);
TEMPLATE_WRAP_FUNC(vertConvolve);
TEMPLATE_WRAP_FUNC(convolve);
TEMPLATE_WRAP_FUNC(gaussianFilter);

TEMPLATE_WRAP_FUNC(drawLine);
TEMPLATE_WRAP_FUNC(drawRectangle);
#ifdef SWIGPYTHON
TEMPLATE_WRAP_FUNC_2T(drawRectangles);
#else
TEMPLATE_WRAP_FUNC_2T_CROSS(drawRectangles);
#endif // SWIGPYTHON
TEMPLATE_WRAP_FUNC(drawBox);
TEMPLATE_WRAP_FUNC(drawCircle);
TEMPLATE_WRAP_FUNC(drawSphere);
TEMPLATE_WRAP_FUNC(drawDisc);
#ifdef USE_FREETYPE
TEMPLATE_WRAP_FUNC(drawText);
#endif // USE_FREETYPE
TEMPLATE_WRAP_FUNC(copyPattern);

TEMPLATE_WRAP_FUNC(vertFlip);
TEMPLATE_WRAP_FUNC(horizFlip);
TEMPLATE_WRAP_FUNC(rotateX90);
TEMPLATE_WRAP_FUNC(translate);
// TEMPLATE_WRAP_FUNC(resizeClosest);
// TEMPLATE_WRAP_FUNC(scaleClosest);
TEMPLATE_WRAP_FUNC(resize);
TEMPLATE_WRAP_FUNC(scale);
TEMPLATE_WRAP_FUNC(addBorder);

// %template(bresenhamLine);

%include "DBlob.hpp"

// workaround for undefined SWIGPY_SLICE_ARG with swig 2.0.3 and 2.0.4
%{
#ifndef SWIGPY_SLICE_ARG
  #if PY_VERSION_HEX >= 0x03020000
    # define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
  #else
    # define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
  #endif
#endif // SWIGPY_SLICE_ARG
%}

namespace std
{
    %template(PixelSequenceVector) vector<PixelSequence>;
    TEMPLATE_WRAP_MAP_FIX_SECOND(Blob, BlobMap);
}

TEMPLATE_WRAP_FUNC(computeBlobs);
TEMPLATE_WRAP_FUNC_2T_CROSS(drawBlobs)
TEMPLATE_WRAP_FUNC(getBlobIDFromOffset);
TEMPLATE_WRAP_FUNC_2T_CROSS(getBlobIDFromOffset);

%include "DMeasures.hpp"
TEMPLATE_WRAP_FUNC(vol);
TEMPLATE_WRAP_FUNC(volume);
%apply double *OUTPUT{double &mean_val};
%apply double *OUTPUT{double &std_dev_val};
TEMPLATE_WRAP_FUNC(meanVal);
TEMPLATE_WRAP_FUNC(area);
TEMPLATE_WRAP_FUNC(minVal);
TEMPLATE_WRAP_FUNC(maxVal);
TEMPLATE_WRAP_FUNC(rangeVal);
TEMPLATE_WRAP_FUNC(valueList);
TEMPLATE_WRAP_FUNC(modeVal);
TEMPLATE_WRAP_FUNC(medianVal);
TEMPLATE_WRAP_FUNC(profile);
TEMPLATE_WRAP_FUNC(measBarycenter);
TEMPLATE_WRAP_FUNC(measBoundBox);
TEMPLATE_WRAP_FUNC(measMoments);
TEMPLATE_WRAP_FUNC(measCovariance);
TEMPLATE_WRAP_FUNC(measAutoCovariance);
// TEMPLATE_WRAP_FUNC(measCenteredCovariance);
// TEMPLATE_WRAP_FUNC(measCenteredAutoCovariance);
TEMPLATE_WRAP_FUNC(measEntropy);
TEMPLATE_WRAP_FUNC(nonZeroOffsets);
TEMPLATE_WRAP_FUNC(isBinary);

// Suppl. Types
TEMPLATE_WRAP_SUPPL_FUNC(vol);
TEMPLATE_WRAP_SUPPL_FUNC(volume);
%apply double *OUTPUT{double &mean_val};
%apply double *OUTPUT{double &std_dev_val};
TEMPLATE_WRAP_SUPPL_FUNC(meanVal);
TEMPLATE_WRAP_SUPPL_FUNC(area);
TEMPLATE_WRAP_SUPPL_FUNC(minVal);
TEMPLATE_WRAP_SUPPL_FUNC(maxVal);
TEMPLATE_WRAP_SUPPL_FUNC(rangeVal);



%include "DBlobMeasures.hpp"
TEMPLATE_WRAP_FUNC(blobsArea);
TEMPLATE_WRAP_FUNC(blobsBarycenter);
TEMPLATE_WRAP_FUNC(blobsBoundBox);
TEMPLATE_WRAP_FUNC(blobsMoments);

// TEMPLATE_WRAP_FUNC(blobsArea);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsMinVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsMaxVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsRangeVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsMeanVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsVolume);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsValueList);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsModeVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsMedianVal);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsBarycenter);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsBoundBox);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsMoments);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsEntropy);

%include "DBlobOperations.hpp"
//TEMPLATE_WRAP_FUNC_2T_CROSS(areaThreshold);
TEMPLATE_WRAP_FUNC(areaThreshold);
TEMPLATE_WRAP_FUNC(blobsInertiaMatrix);
TEMPLATE_WRAP_FUNC_2T_CROSS(blobsInertiaMatrix);

%include "DImageMatrix.hpp"
TEMPLATE_WRAP_FUNC(matMultiply);
TEMPLATE_WRAP_FUNC(matTranspose);

%include "DImageCompare.hpp"
TEMPLATE_WRAP_FUNC(indexJaccard);
TEMPLATE_WRAP_FUNC(indexRuzicka);
TEMPLATE_WRAP_FUNC(indexAccuracy);
TEMPLATE_WRAP_FUNC(indexPrecision);
TEMPLATE_WRAP_FUNC(indexRecall);
TEMPLATE_WRAP_FUNC(indexFScore);
TEMPLATE_WRAP_FUNC(indexSensitivity);
TEMPLATE_WRAP_FUNC(indexSpecificity);
TEMPLATE_WRAP_FUNC(indexFallOut);
TEMPLATE_WRAP_FUNC(indexMissRate);
TEMPLATE_WRAP_FUNC(indexOverlap);
TEMPLATE_WRAP_FUNC(distanceHamming);
TEMPLATE_WRAP_FUNC(distanceHausdorff);

