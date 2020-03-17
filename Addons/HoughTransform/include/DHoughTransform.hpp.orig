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


#ifndef _D_HOUGH_TRANSFORM_HPP
#define _D_HOUGH_TRANSFORM_HPP

#include "Core/include/private/DImage.hxx"

namespace smil
{
    /**
     * @ingroup Addons
     * @addtogroup AddonHoughTransform Features detection (Hough)
     * @{
     */

    /**
     * Hough Lines
     *
     */	
    template <class T1, class T2>
    RES_T houghLines(Image<T1> &imIn, double thetaRes, double rhoRes, Image<T2> &imOut)
    {
        UINT wIn = imIn.getWidth();
        UINT hIn = imIn.getHeight();
        
        double rhoMax = sqrt(wIn*wIn + hIn*hIn);
        
        UINT wOut = thetaRes * 180;
        UINT hOut = rhoRes * rhoMax;
        
        
        ImageFreezer freeze(imOut);
        imOut.setSize(wOut, hOut);
        fill(imOut, T2(0));
        
        typename Image<T1>::sliceType linesIn = imIn.getSlices()[0];
        typename Image<T1>::lineType lIn;
        typename Image<T2>::sliceType linesOut = imOut.getSlices()[0];
        
        double thetaStep = PI / double(wOut);
        double rhoStep = rhoMax / double(hOut);
        double theta;
        UINT rho;
        
        for (UINT j=0;j<imIn.getHeight();j++)
        {
            lIn = linesIn[j];
            for (UINT i=0;i<imIn.getWidth();i++)
            {
                if (lIn[i]!=0)
                {
                  for (UINT t=0;t<wOut;t++)
                  {
                      theta = t*thetaStep;
                      rho = (i*cos(theta) + j*sin(theta)) / rhoStep;
                      if (rho<hOut)
                        linesOut[rho][t] += 1;
                  }
                    
                }
            }
        }
	return RES_OK;
    }
    
    /**
     * Hough Circles
     *
     */	
    template <class T1, class T2>
    RES_T houghCircles(Image<T1> &imIn, double xyResol, double rhoResol, Image<T2> &imOut)
    {
        UINT wIn = imIn.getWidth();
        UINT hIn = imIn.getHeight();
        
        double rhoMax = sqrt(wIn*wIn + hIn*hIn);
        
        UINT wOut = xyResol * wIn;
        UINT hOut = xyResol * hIn;
        UINT dOut = rhoResol * rhoMax;
        
        ImageFreezer freeze(imOut);
        imOut.setSize(wOut, hOut, dOut);
        fill(imOut, T2(0));
        
        typename Image<T1>::sliceType linesIn = imIn.getSlices()[0];
        typename Image<T1>::lineType lIn;
        typename Image<T2>::volType slicesOut = imOut.getSlices();
        
        UINT rho;
        
        for (UINT j=0;j<imIn.getHeight();j++)
        {
            lIn = linesIn[j];
            for (UINT i=0;i<imIn.getWidth();i++)
            {
                if (lIn[i]!=T1(0))
                {
                  for (UINT j2=0;j2<hOut;j2++)
                    for (UINT i2=0;i2<wOut;i2++)
                      if (i!=i2 && j!=j2)
                    {
                        rho = sqrt(double((i*xyResol-i2)*(i*xyResol-i2)+(j*xyResol-j2)*(j*xyResol-j2)))/xyResol*rhoResol;
                        if (rho<dOut)
                          slicesOut[rho][j2][i2] += 1;
                    }
                    
                }
            }
        }
        return RES_OK;
    }
    
    /**
     * Hough Circles
     *
     */	
    template <class T1, class T2>
    RES_T houghCircles(Image<T1> &imIn, double resol, Image<T2> &imOut, Image<T2> &imRadiiOut)
    {
        UINT wIn = imIn.getWidth();
        UINT hIn = imIn.getHeight();
       
        // Commente - rhoMax non utilise	
        // double rhoMax = sqrt(wIn*wIn + hIn*hIn);
        
        UINT wOut = resol * wIn;
        UINT hOut = resol * hIn;
        
        ImageFreezer freeze(imRadiiOut);
        
        Image<double> imCentersOut(wOut, hOut);
        imRadiiOut.setSize(wOut, hOut);
        
        fill(imCentersOut, 0.);
        fill(imRadiiOut, T2(0));
        
        typename Image<T1>::sliceType linesIn = imIn.getSlices()[0];
        typename Image<T1>::lineType lIn;
        typename Image<double>::sliceType linesOut1 = imCentersOut.getSlices()[0];
        typename Image<T2>::sliceType linesOut2 = imRadiiOut.getSlices()[0];
        
        double rho;
        UINT nonZeroPts = 0;
        
        for (UINT j=0;j<imIn.getHeight();j++)
        {
            lIn = linesIn[j];
            for (UINT i=0;i<imIn.getWidth();i++)
            {
                if (lIn[i]!=T1(0))
                {
                  nonZeroPts++;
                  for (UINT j2=0;j2<hOut;j2++)
                    for (UINT i2=0;i2<wOut;i2++)
                       if (i!=i2 && j!=j2)
                    {
                        rho = sqrt(double((i*resol-i2)*(i*resol-i2)+(j*resol-j2)*(j*resol-j2)));
                        if (rho > 100 && rho<500)
                        {
                            linesOut1[j2][i2] += 1;
                            if (linesOut1[j2][i2]>nonZeroPts && linesOut2[j2][i2]<T2(rho))
                              linesOut2[j2][i2] = T2(rho);
                        }
                    }
                    
                }
            }
        }
        stretchHist(imCentersOut, imOut);
        return RES_OK;    
    }
    
   /**@}*/ 
} // namespace smil

#endif // _D_HOUGH_TRANSFORM_HPP

 
