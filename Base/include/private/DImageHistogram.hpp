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


#ifndef _D_IMAGE_HISTOGRAM_HPP
#define _D_IMAGE_HISTOGRAM_HPP

#include <limits.h>

#include "DLineHistogram.hpp"
#include "DImageArith.hpp"

    
namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Histogram
    * \{
    */

#ifndef SWIG
    template <class T>
    RES_T histogram(const Image<T> &imIn, size_t *h)
    {
        for (int i=int(ImDtTypes<T>::min());i<=int(ImDtTypes<T>::max());i++)
            h[i] = 0;

        typename Image<T>::lineType pixels = imIn.getPixels();
        for (size_t i=0;i<imIn.getPixelCount();i++)
            h[UINT(pixels[i]-ImDtTypes<T>::min())]++;
        
        return RES_OK;
    }
#endif // SWIG

    /**
    * Image histogram
    */
    template <class T>
    std::map<T, UINT> histogram(const Image<T> &imIn)
    {
        size_t *buf = new size_t[ImDtTypes<T>::cardinal()];
        histogram<T>(imIn, buf);
        
        map<T, UINT> h;
        for (size_t i=ImDtTypes<T>::min();i<=ImDtTypes<T>::max();i++)
            h.insert(pair<T,UINT>(i, buf[i]));
        
        delete[] buf;
        
        return h;
    }


    /**
    * Image histogram with a mask image.
    * 
    * Calculates the histogram of the image imIn only for pixels x where imMask(x)!=0
    */
    template <class T>
    std::map<T, UINT> histogram(const Image<T> &imIn, const Image<T> &imMask)
    {
        map<T, UINT> h;
        
        for (T i=ImDtTypes<T>::min();;i++)
        {
            h.insert(pair<T,UINT>(i, 0));
            if (i==ImDtTypes<T>::max())
              break;
        }
        
        typename Image<T>::lineType inPix = imIn.getPixels();
        typename Image<T>::lineType maskPix = imMask.getPixels();
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
            if (maskPix[i]!=0)
                h[T(inPix[i])] += 1;
        
        return h;
    }

    /**
    * Image threshold
    */
    template <class T>
    RES_T threshold(const Image<T> &imIn, T minVal, T maxVal, T trueVal, T falseVal, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        unaryImageFunction<T, threshLine<T> > iFunc;
        
        iFunc.lineFunction.minVal = minVal;
        iFunc.lineFunction.maxVal = maxVal;
        iFunc.lineFunction.trueVal = trueVal;
        iFunc.lineFunction.falseVal = falseVal;
        
        return iFunc(imIn, imOut);
    }

    template <class T>
    RES_T threshold(const Image<T> &imIn, T minVal, T maxVal, Image<T> &imOut)
    {
        return threshold<T>(imIn, minVal, maxVal, ImDtTypes<T>::max(), ImDtTypes<T>::min(), imOut);
    }

    template <class T>
    RES_T threshold(const Image<T> &imIn, T minVal, Image<T> &imOut)
    {
        return threshold<T>(imIn, minVal, ImDtTypes<T>::max(), ImDtTypes<T>::max(), ImDtTypes<T>::min(), imOut);
    }

    template <class T>
    RES_T threshold(const Image<T> &imIn, Image<T> &imOut)
    {
        T tVal = otsuThreshold(imIn, imOut);
        if (tVal==ImDtTypes<T>::min())
          return RES_ERR;
        else
          return RES_OK;
    }

    /**
    * Stretch histogram
    */
    template <class T1, class T2>
    RES_T stretchHist(const Image<T1> &imIn, T1 inMinVal, T1 inMaxVal, Image<T2> &imOut, T2 outMinVal=numeric_limits<T2>::min(), T2 outMaxVal=numeric_limits<T2>::max())
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        unaryImageFunction<T2, stretchHistLine<T2> > iFunc;
        iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (inMaxVal-inMinVal);
        iFunc.lineFunction.inOrig = inMinVal;
        iFunc.lineFunction.outOrig = outMinVal;
        
        return iFunc(imIn, imOut);
    }

    template <class T1, class T2>
    RES_T stretchHist(const Image<T1> &imIn, Image<T2> &imOut, T2 outMinVal, T2 outMaxVal)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        unaryImageFunction<T2, stretchHistLine<T2> > iFunc;
        vector<T1> rangeV = rangeVal(imIn);
        iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (rangeV[1]-rangeV[0]);
        iFunc.lineFunction.inOrig = rangeV[0];
        iFunc.lineFunction.outOrig = outMinVal;
        
        return iFunc(imIn, imOut);
    }

    template <class T1, class T2>
    RES_T stretchHist(const Image<T1> &imIn, Image<T2> &imOut)
    {
        return stretchHist<T1,T2>(imIn, imOut, numeric_limits<T2>::min(), numeric_limits<T2>::max());
    }


    /**
    * Enhance contrast
    */
    template <class T>
    RES_T enhanceContrast(const Image<T> &imIn, Image<T> &imOut, double leftSat, double rightSat)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        size_t *h = new size_t[ImDtTypes<T>::cardinal()];
        histogram(imIn, h);
        
        double imVol = imIn.getPixelCount();
        double satVol;
        double curVol;
        vector<T> rangeV = rangeVal(imIn);
        T threshValLeft = rangeV[0];
        T threshValRight = rangeV[1];
        
        // left
        satVol = imVol * leftSat / 100.;
        curVol=0;
        for (size_t i=rangeV[0]; i<rangeV[1]; i++)
        {
            curVol += double(h[i]);
            if (curVol>satVol)
              break;
            threshValLeft = i;
        }
        
        // Right
        satVol = imVol * rightSat / 100.;
        curVol=0;
        for (size_t i=rangeV[1]; i>size_t(rangeV[0]); i--)
        {
            curVol += h[i];
            if (curVol>satVol)
              break;
            threshValRight = i;
        }
        
        stretchHist(imIn, threshValLeft, threshValRight, imOut);
        imOut.modified();
        
        delete[] h;
        
        return RES_OK;
    }

    template <class T>
    RES_T enhanceContrast(const Image<T> &imIn, Image<T> &imOut, double sat=0.5)
    {
        return enhanceContrast(imIn, imOut, sat/2., sat/2.);
    }

    template <class T>
    bool IncrementThresholds(vector<double> &thresholdIndexes, map<T, UINT> &hist, UINT threshLevels, double totalFrequency, double &globalMean, vector<double> &classMean, vector<double> &classFrequency)
    {
      unsigned long numberOfHistogramBins = hist.size();
      unsigned long numberOfClasses = classMean.size();

      typedef double MeanType;
      typedef double FrequencyType;
      
      MeanType meanOld;
      FrequencyType freqOld;

      unsigned int k;
      int j;

      // from the upper threshold down
      for(j=static_cast<int>(threshLevels-1); j>=0; j--)
        {
        // if this threshold can be incremented (i.e. we're not at the end of the histogram)
        if (thresholdIndexes[j] < numberOfHistogramBins - 2 - (threshLevels-1 - j) )
          {
          // increment it and update mean and frequency of the class bounded by the threshold
          thresholdIndexes[j] += 1;

          meanOld = classMean[j];
          freqOld = classFrequency[j];
          
          classFrequency[j] += hist[UINT(thresholdIndexes[j])];
          
          if (classFrequency[j]>0)
            {
            classMean[j] = (meanOld * static_cast<MeanType>(freqOld)
                            + static_cast<MeanType>(thresholdIndexes[j])
                            * static_cast<MeanType>(hist[UINT(thresholdIndexes[j])]))
              / static_cast<MeanType>(classFrequency[j]);
            }
          else
            {
            classMean[j] = 0;
            }
          
          // set higher thresholds adjacent to their previous ones, and update mean and frequency of the respective classes
          for (k=j+1; k<threshLevels; k++)
            {
            thresholdIndexes[k] = thresholdIndexes[k-1] + 1;
            classFrequency[k] = hist[UINT(thresholdIndexes[k])];
            if (classFrequency[k]>0)
              {
              classMean[k] = static_cast<MeanType>(thresholdIndexes[k]);
              }
            else
              {
              classMean[k] = 0;
              }
            }
          
          // update mean and frequency of the highest class
          classFrequency[numberOfClasses-1] = totalFrequency;
          classMean[numberOfClasses-1] = globalMean * totalFrequency;

          for(k=0; k<numberOfClasses-1; k++)
            {
            classFrequency[numberOfClasses-1] -= classFrequency[k];
            classMean[numberOfClasses-1] -= classMean[k] * static_cast<MeanType>(classFrequency[k]);
            }

          if (classFrequency[numberOfClasses-1]>0)
            {
            classMean[numberOfClasses-1] /= static_cast<MeanType>(classFrequency[numberOfClasses-1]);
            }
          else
            {
            classMean[numberOfClasses-1] = 0;
            }

          // exit the for loop if a threshold has been incremented
          break;
          }
        else  // if this threshold can't be incremented
          {
          // if it's the lowest threshold
          if (j==0)
            {
            // we couldn't increment because we're done
            return false;
            }
          }
        }
      // we incremented
      return true;
    }

    template <class T>
    vector<T> otsuThresholdValues(map<T, UINT> &hist, UINT threshLevels=1)
    {
        
        typedef double MeanType;
        typedef vector<MeanType> MeanVectorType;

      
        double totalFrequency = 0;
        MeanType globalMean = 0;
        
        for (typename map<T, UINT>::iterator it=hist.begin();it!=hist.end();it++)
        {
            globalMean += double((*it).first) * double((*it).second);
            totalFrequency += (*it).second;
        }
        
        globalMean /= totalFrequency;

        
        unsigned long numberOfClasses = threshLevels + 1;
        MeanVectorType thresholdIndexes(threshLevels, 0);

        for(unsigned long j=0; j<threshLevels; j++)
        {
            thresholdIndexes[j] = j;
        }

        MeanVectorType maxVarThresholdIndexes = thresholdIndexes;
        double freqSum = 0;
        MeanVectorType classFrequency(numberOfClasses, 0);
        
        for (unsigned long j=0; j<numberOfClasses-1; j++)
        {
            classFrequency[j] = hist[T(thresholdIndexes[j])];
            freqSum += classFrequency[j];
        }
        
        classFrequency[numberOfClasses-1] = totalFrequency - freqSum;
      
        double meanSum = 0;
        MeanVectorType classMean(numberOfClasses, 0);
        
        for (unsigned long j=0; j < numberOfClasses-1; j++)
        {
          if (classFrequency[j]>0)
          {
              classMean[j] = j;
          }
          else
          {
              classMean[j] = 0;
          }
          meanSum += classMean[j] * classFrequency[j];
        }

        if (classFrequency[numberOfClasses-1]>0)
        {
            classMean[numberOfClasses-1] = double(globalMean * totalFrequency - meanSum) / double(classFrequency[numberOfClasses-1]);
        }
        else
        {
            classMean[numberOfClasses-1] = 0;
        }
      
        double maxVarBetween = 0;
        for (unsigned long j=0; j<numberOfClasses; j++)
        {
            maxVarBetween += classFrequency[j] * (globalMean - classMean[j]) * (globalMean - classMean[j]);
        }

        // explore all possible threshold configurations and choose the one that yields maximum between-class variance
        while (IncrementThresholds(thresholdIndexes, hist, threshLevels, totalFrequency, globalMean, classMean, classFrequency))
        {
            double varBetween = 0;
            for (unsigned long j=0; j<numberOfClasses; j++)
            {
                varBetween += classFrequency[j] * (globalMean - classMean[j]) * (globalMean - classMean[j]);
            }

            if (varBetween > maxVarBetween)
            {
                maxVarBetween = varBetween;
                maxVarThresholdIndexes = thresholdIndexes;
            }
        }

        vector<T> threshVals;
        
        for (unsigned long j=0; j<threshLevels; j++)
        {
            threshVals.push_back(T(maxVarThresholdIndexes[j])); //= histogram->GetBinMax(0,maxVarThresholdIndexes[j]);
        }
        
        return threshVals;
    }

    template <class T>
    vector<T> otsuThresholdValues(const Image<T> &im, UINT threshLevels=1)
    {
        map<T, UINT> hist = histogram(im);
        return otsuThresholdValues(hist, threshLevels);
    }

    template <class T>
    vector<T> otsuThresholdValues(const Image<T> &im, const Image<T> &imMask, UINT threshLevels=1)
    {
        map<T, UINT> hist = histogram(im, imMask);
        return otsuThresholdValues(hist, threshLevels);
    }


    /**
    * Otsu Threshold
    * 
    * \demo{thresholds.py}
    */
    template <class T>
    vector<T> otsuThreshold(const Image<T> &imIn, Image<T> &imOut, UINT nbrThresholds)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
          return vector<T>();
        
        vector<T> tVals = otsuThresholdValues<T>(imIn, nbrThresholds);
        map<T, T> lut;
        T i = ImDtTypes<T>::min();
        T lbl = 0;
        for (typename vector<T>::iterator it=tVals.begin();it!=tVals.end();it++,lbl++)
        {
            while(i<(*it))
            {
                lut[i] = lbl;
                i++;
            }
        }
        while(i<=ImDtTypes<T>::max() && i>ImDtTypes<T>::min())
        {
            lut[i] = lbl;
            i++;
        }
        applyLookup(imIn, lut, imOut);
        
        return tVals;
        
    }

    template <class T>
    T otsuThreshold(const Image<T> &imIn, Image<T> &imOut)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
          return ImDtTypes<T>::min();
        
        vector<T> tVals = otsuThresholdValues<T>(imIn, 1);
        threshold<T>(imIn, tVals[0], imOut);
        return tVals[0];
    }


    template <class T>
    vector<T> otsuThreshold(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, UINT nbrThresholds=1)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
          return vector<T>();
        
        vector<T> tVals = otsuThresholdValues<T>(imIn, imMask, nbrThresholds);
        threshold<T>(imIn, tVals[0], imOut);
        
        return tVals;
        
    }

/** \} */

} // namespace smil


#endif // _D_IMAGE_HISTOGRAM_HPP

