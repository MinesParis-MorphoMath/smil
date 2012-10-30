/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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

#include "DLineHistogram.hpp"
#include "DImageArith.hpp"

/**
 * \ingroup Base
 * \defgroup Histogram
 * \{
 */

/**
 * Image histogram
 */
template <class T>
map<T, UINT> histogram(const Image<T> &imIn)
{
    map<T, UINT> h;
    for (T i=ImDtTypes<T>::min();i<ImDtTypes<T>::max();i+=1)
      h.insert(pair<T,UINT>(i, 0));
    
    typename Image<T>::lineType pixels = imIn.getPixels();
    for (UINT i=0;i<imIn.getPixelCount();i++)
	h[pixels[i]] += 1;
    
    return h;
}

/**
 * Image histogram with a mask image
 */
template <class T>
map<T, UINT> histogram(const Image<T> &imIn, const Image<T> &imMask)
{
    map<T, UINT> h;
    
    for (T i=ImDtTypes<T>::min();i<ImDtTypes<T>::max();i+=1)
      h.insert(pair<T,UINT>(i, 0));
    
    typename Image<T>::lineType inPix = imIn.getPixels();
    typename Image<T>::lineType maskPix = imMask.getPixels();
    
    for (UINT i=0;i<imIn.getPixelCount();i++)
	if (maskPix[i]!=0)
	    h[inPix[i]] += 1;
    
    return h;
}

/**
 * Image threshold
 */
template <class T>
RES_T threshold(const Image<T> &imIn, T minVal, T maxVal, T trueVal, T falseVal, Image<T> &imOut)
{
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
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
    return iFunc(imIn, imOut);
}

template <class T>
RES_T threshold(const Image<T> &imIn, T minVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = numeric_limits<T>::max();
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
    return iFunc(imIn, imOut);
}

/**
 * Stretch histogram
 */
template <class T1, class T2>
RES_T stretchHist(const Image<T1> &imIn, T1 inMinVal, T1 inMaxVal, Image<T2> &imOut, T2 outMinVal=numeric_limits<T2>::min(), T2 outMaxVal=numeric_limits<T2>::max())
{
    unaryImageFunction<T2, stretchHistLine<T2> > iFunc;
    iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (inMaxVal-inMinVal);
    iFunc.lineFunction.inOrig = inMinVal;
    iFunc.lineFunction.outOrig = outMinVal;
    
    return iFunc(imIn, imOut);
}

template <class T1, class T2>
RES_T stretchHist(const Image<T1> &imIn, Image<T2> &imOut, T2 outMinVal, T2 outMaxVal)
{
    unaryImageFunction<T2, stretchHistLine<T2> > iFunc;
    T1 rmin, rmax;
    rangeVal(imIn, &rmin, &rmax);
    iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (rmax-rmin);
    iFunc.lineFunction.inOrig = rmin;
    iFunc.lineFunction.outOrig = outMinVal;
    
    return iFunc(imIn, imOut);
}

template <class T1, class T2>
RES_T stretchHist(const Image<T1> &imIn, Image<T2> &imOut)
{
    Image<T1> tmpIm(imIn);
    RES_T res = stretchHist<T1>(imIn, tmpIm, (T1)numeric_limits<T2>::min(), (T1)numeric_limits<T2>::max());
    if (res!=RES_OK)
      return res;
    return copy(tmpIm, imOut);
}


/**
 * Enhance contrast
 */
template <class T>
RES_T enhanceContrast(const Image<T> &imIn, Image<T> &imOut, double sat=0.5)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;
    
    map<T, UINT> h = histogram(imIn);
    double imVol = imIn.getWidth() * imIn.getHeight() * imIn.getDepth();
    double satVol = imVol * sat / 100.;
    double v = 0;
    T minV, maxV, threshVal;
    rangeVal(imIn, &minV, &maxV);
    
    for (T i=maxV; i>=minV; i-=1)
    {
	v += h[i];
	if (v>satVol)
	    break;
	threshVal = i;
    }
    
    stretchHist(imIn, minV, threshVal, imOut);
    imOut.modified();
    
    return RES_OK;
}

/** \} */


/**
 * Otsu
 */

template <class T>
bool IncrementThresholds(vector<double> &thresholdIndexes, map<T, UINT> &hist, UINT threshLevels, double totalFrequency, double &globalMean, vector<double> &classMean, vector<double> &classFrequency)
{
  
    typedef double MeanType;
  
    unsigned long numberOfHistogramBins = hist.size();
    unsigned long numberOfClasses = classMean.size();

    MeanType meanOld;
    MeanType freqOld;

    unsigned int k;
    int j;

    // from the upper threshold down
    for(j=threshLevels-1; j>=0; j--)
    {
      // if this threshold can be incremented (i.e. we're not at the end of the histogram)
      if (thresholdIndexes[j] < numberOfHistogramBins - 2 - (threshLevels-1 - j) )
      {
	// increment it and update mean and frequency of the class bounded by the threshold
	thresholdIndexes[j] += 1;

	meanOld = classMean[j];
	freqOld = classFrequency[j];
      
	classFrequency[j] += hist[thresholdIndexes[j]];
      
	if (classFrequency[j]>0)
        {
	    classMean[j] = double(meanOld * freqOld + thresholdIndexes[j] * hist[thresholdIndexes[j]] ) / double(classFrequency[j]);
        }
	else
        {
	    classMean[j] = 0;
        }
      
      // set higher thresholds adjacent to their previous ones, and update mean and frequency of the respective classes
      for (k=j+1; k<threshLevels; k++)
      {
	  thresholdIndexes[k] = thresholdIndexes[k-1] + 1;
	  classFrequency[k] = hist[thresholdIndexes[k]];
	  if (classFrequency[k]>0)
          {
	      classMean[k] = k;
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
        classMean[numberOfClasses-1] -= classMean[k] * classFrequency[k];
        }

	if (classFrequency[numberOfClasses-1]>0)
        {
	    classMean[numberOfClasses-1] = double(classMean[numberOfClasses-1])/double(classFrequency[numberOfClasses-1]);
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
vector<T> otsuThresholdValues(map<T, UINT> &hist, UINT threshLevels=2)
{
    
    typedef double MeanType;
    typedef vector<MeanType> MeanVectorType;

  
    double totalFrequency = 0;
    MeanType globalMean = 0;
    
    for (typename map<T, UINT>::iterator it=hist.begin();it!=hist.end();it++)
    {
	globalMean += (*it).first * (*it).second;
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
	classFrequency[j] = hist[thresholdIndexes[j]];
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
	threshVals.push_back(maxVarThresholdIndexes[j]); //= histogram->GetBinMax(0,maxVarThresholdIndexes[j]);
    }
    
    return threshVals;
}

template <class T>
vector<T> otsuThresholdValues(const Image<T> &im, UINT threshLevels=2)
{
    map<T, UINT> hist = histogram(im);
    return otsuThresholdValues(hist, threshLevels);
}

template <class T>
vector<T> otsuThresholdValues(const Image<T> &im, const Image<T> &imMask, UINT threshLevels=2)
{
    map<T, UINT> hist = histogram(im, imMask);
    return otsuThresholdValues(hist, threshLevels);
}



template <class T>
RES_T otsuThreshold(const Image<T> &imIn, Image<T> &imOut, UINT nbrThresholds=1)
{
    ASSERT_ALLOCATED(&imIn, &imOut);
    
    vector<T> tVals = otsuThresholdValues<T>(imIn, nbrThresholds);
    threshold<T>(imIn, tVals[0], imOut);
    
    return RES_OK;
    
}

template <class T>
RES_T otsuThreshold(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, UINT nbrThresholds=1)
{
    ASSERT_ALLOCATED(&imIn, &imOut);
    
    vector<T> tVals = otsuThresholdValues<T>(imIn, imMask, nbrThresholds);
    threshold<T>(imIn, tVals[0], imOut);
    
    return RES_OK;
    
}

#endif // _D_IMAGE_HISTOGRAM_HPP

