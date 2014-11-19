/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#include "DColorConvert.h"

#include <complex>

namespace smil
{
    RES_T RGBToXYZ(const Image<RGB> &imRgbIn, Image<RGB> &imXyzOut)
    {
        ASSERT_ALLOCATED(&imRgbIn, &imXyzOut);
        ASSERT_SAME_SIZE(&imRgbIn, &imXyzOut);
        
        static const float yc1 = 0.607f,       yc2 = 0.174f,   yc3 = 0.201f;
        static const float uc1 = 0.299f,       uc2 = 0.587f,   uc3 = 0.114f;
        static const float vc1 = 0.000f,       vc2 = 0.066f,   vc3 = 1.117f;
        
        Image<UINT8>::lineType r = imRgbIn.getPixels().arrays[0];
        Image<UINT8>::lineType g = imRgbIn.getPixels().arrays[1];
        Image<UINT8>::lineType b = imRgbIn.getPixels().arrays[2];
        
        Image<UINT8>::lineType x = imXyzOut.getPixels().arrays[0];
        Image<UINT8>::lineType y = imXyzOut.getPixels().arrays[1];
        Image<UINT8>::lineType z = imXyzOut.getPixels().arrays[2];
        
      #pragma omp for
        for (size_t i=0;i<imRgbIn.getPixelCount();i++)
        {
            x[i] = UINT8(floor(0.5+(yc1 * r[i] + yc2 * g[i] + yc3 * b[i])/0.982));
            y[i] = UINT8(floor(0.5+uc1 * r[i] + uc2 * g[i] + uc3 * b[i] )); 
            z[i] = UINT8(floor(0.5+(vc1 * r[i] + vc2 * g[i] + vc3 * b[i])/1.183));
        }
        

        imXyzOut.modified();
        
        return RES_OK;
      
    }
    
    RES_T XYZToRGB(const Image<RGB> &imXyzIn, Image<RGB> &imRgbOut)
    {
        ASSERT_ALLOCATED(&imXyzIn, &imRgbOut);
        ASSERT_SAME_SIZE(&imXyzIn, &imRgbOut);
        
        Image<UINT8>::lineType x = imXyzIn.getPixels().arrays[0];
        Image<UINT8>::lineType y = imXyzIn.getPixels().arrays[1];
        Image<UINT8>::lineType z = imXyzIn.getPixels().arrays[2];
        
        Image<UINT8>::lineType r = imRgbOut.getPixels().arrays[0];
        Image<UINT8>::lineType g = imRgbOut.getPixels().arrays[1];
        Image<UINT8>::lineType b = imRgbOut.getPixels().arrays[2];
        
      #pragma omp for
        for (size_t i=0;i<imXyzIn.getPixelCount();i++)
        {
            float Rf, Gf,Bf;

            /* X, Y, Z are extracted */
            UINT8 X = static_cast<float> (x[i]);
            UINT8 Y = static_cast<float> (y[i]);
            UINT8 Z = static_cast<float> (z[i]);

            /* calculus of R, G, B */
            Rf =  (1.910364 * 0.982 * X )-(0.533748 * Y) -
            (0.289289 * 1.183 * Z) ;
            Gf = (-0.984377 * 0.982 * X) +
            (1.998384 * Y) -
            0.026818 * 1.183 * Z ;
            Bf =  (0.058164 * 0.982 * X) - (0.118078 * Y )+
            (0.896840 * 1.183 * Z) ;

            /* test to avoid values outside [0,255] */
            if (Rf < 0) Rf = 0;
            else {
            if (Rf > 255) Rf= 255;
            }

            if (Gf < 0) Gf = 0;
            else {
            if (Gf > 255) Gf =255;
            }

            if (Bf < 0) Bf = 0;
            else {
            if (Bf > 255)Bf=255;
            }

            /* rescaling and normalizing of output values */      
            r[i] = (UINT8)Rf;
            g[i] =  (UINT8)Gf;
            b[i] =  (UINT8)Bf;
        }

        imRgbOut.modified();
        
        return RES_OK;
      
    }
    
    inline float scale_from_255(const float &x, const float &xmin, const float &xmax)
    {
        return (float) ((xmin)+  (x) * ((xmax) - (xmin))/ 255.0 );
    }
    
    inline UINT8 scale_to_255(const float &x, const float &xmin, const float &xmax)
    {
        UINT8 retVal;
        if (x<xmin)
          retVal = 0;
        else if (x>xmax)
          retVal = 255;
        else
          retVal = floor(0.5+ ( 255.0 * ((x) - (xmin)) / ((xmax) - (xmin)) ));

        return retVal;
    }
    inline float lab_conditional(const float &x)
    {
        if( x < 0.008856) 
          return 7.787*x+16/116.0;
        else
          return std::pow (static_cast<double>(x),(1/3.0));
    }
    
    RES_T XYZToLAB(const Image<RGB> &imXyzIn, Image<RGB> &imLabOut)
    {
        ASSERT_ALLOCATED(&imXyzIn, &imLabOut);
        ASSERT_SAME_SIZE(&imXyzIn, &imLabOut);
        
        static const float yc1 = 0.607f,       yc2 = 0.174f,   yc3 = 0.201f;
        static const float uc1 = 0.299f,       uc2 = 0.587f,   uc3 = 0.114f;
        static const float vc1 = 0.000f,       vc2 = 0.066f,   vc3 = 1.117f;
        
        Image<UINT8>::lineType x = imXyzIn.getPixels().arrays[0];
        Image<UINT8>::lineType y = imXyzIn.getPixels().arrays[1];
        Image<UINT8>::lineType z = imXyzIn.getPixels().arrays[2];
        
        Image<UINT8>::lineType l = imLabOut.getPixels().arrays[0];
        Image<UINT8>::lineType a = imLabOut.getPixels().arrays[1];
        Image<UINT8>::lineType b = imLabOut.getPixels().arrays[2];

      
      #pragma omp for
        for (size_t i=0;i<imXyzIn.getPixelCount();i++)
        {
            float L, A, B;
            
            float Xf = scale_from_255( x[i], 0.0, 0.982) ;
            float Yf = scale_from_255( y[i], 0.0, 1.0) ;
            float Zf = scale_from_255( z[i], 0.0, 1.183) ;
            
            if (Yf >= 0.008856) 
              L = (float) (-16.0 + 25.0 * std::pow (static_cast<double>(100 * Yf) ,(1/3.0))) ;
            else
              L = (float) (903.3 * Yf) ;
            
            float x_modified = lab_conditional(Xf/0.982);
            float y_modified = lab_conditional(Yf);
            float z_modified = lab_conditional(Zf/1.183);

            A = 500.0 * (x_modified - y_modified);
            B = 200.0 * (y_modified - z_modified);

            /* rescaling and normalizing of output values */          
            l[i] = scale_to_255(L, 0.0, 100.0397);
            a[i] = scale_to_255(A, -137.8146, 96.1775);
            b[i] = scale_to_255(B, -99.2331, 115.6697);
        }
        

        imLabOut.modified();
        
        return RES_OK;
      
    }
    
    RES_T LABToXYZ(const Image<RGB> &imLabIn, Image<RGB> &imXyzOut)
    {
        ASSERT_ALLOCATED(&imLabIn, &imXyzOut);
        ASSERT_SAME_SIZE(&imLabIn, &imXyzOut);
        
        Image<UINT8>::lineType l = imLabIn.getPixels().arrays[0];
        Image<UINT8>::lineType a = imLabIn.getPixels().arrays[1];
        Image<UINT8>::lineType b = imLabIn.getPixels().arrays[2];

        Image<UINT8>::lineType x = imXyzOut.getPixels().arrays[0];
        Image<UINT8>::lineType y = imXyzOut.getPixels().arrays[1];
        Image<UINT8>::lineType z = imXyzOut.getPixels().arrays[2];
        
      
      #pragma omp for
        for (size_t i=0;i<imLabIn.getPixelCount();i++)
        {
            float X, Y, Z;
            
            UINT8 L = static_cast<float> (l[i]);
            UINT8 A = static_cast<float> (a[i]);
            UINT8 B = static_cast<float> (b[i]);

            float Lf = (float) scale_from_255( L, 0.0, 100.0397);
            float Af = (float) scale_from_255( A, -137.8146, 96.1775) ;      
            float Bf = (float) scale_from_255(B, -99.2331, 115.6697) ;
            

            /* calculus of Y */
            if (Lf >= 7.996248)
                Y = (float) (::powf ( ((Lf + 16) / 25) , 3.0) / 100) ;
            else
                Y = (float) (Lf / 903.3) ;

            /* calculus of X and Z */
            X = (float) (0.982 * ::powf( ::powf( Y, 1/3.0) + (Af / 500), 3.0));
            Z = (float) (1.183 * ::powf( ::powf( Y, 1/3.0) - (Bf / 200), 3.0));

            /* rescaling and normalizing of output values */      
            x[i] = scale_to_255(X, 0.0, 0.982);
            y[i] = scale_to_255(Y, - 0.0, 1.0);
            z[i] = scale_to_255(Z,  0.0, 1.183);

        }
        

        imXyzOut.modified();
        
        return RES_OK;
      
    }
    
    RES_T RGBToHLS(const Image<RGB> &imRgbIn, Image<RGB> &imHlsOut)
    {
        ASSERT_ALLOCATED(&imRgbIn, &imHlsOut);
        ASSERT_SAME_SIZE(&imRgbIn, &imHlsOut);
        
        Image<UINT8>::lineType r = imRgbIn.getPixels().arrays[0];
        Image<UINT8>::lineType g = imRgbIn.getPixels().arrays[1];
        Image<UINT8>::lineType b = imRgbIn.getPixels().arrays[2];
        
        Image<UINT8>::lineType h = imHlsOut.getPixels().arrays[0];
        Image<UINT8>::lineType l = imHlsOut.getPixels().arrays[1];
        Image<UINT8>::lineType s = imHlsOut.getPixels().arrays[2];
        
      #pragma omp for
        for (size_t i=0;i<imRgbIn.getPixelCount();i++)
        {
            /* R, G, B are extracted */
            float R = static_cast<float> (r[i]);
            float G = static_cast<float> (g[i]);
            float B = static_cast<float> (b[i]);

            //      tR = static_cast< typename value_type::value_type> (floor(0.5+tmp));

            double mymax,mymin,Mid;
            double L1re,Hre,S1re;
            double   L1ou,Hou,S1ou;
            double   lambda,phi,k;
            double  Rf,Gf,Bf;

            Rf=((double)(R)/255.0)*(1.0);
            Gf=((double)(G)/255.0)*(1.0);
            Bf=((double)(B)/255.0)*(1.0);
            mymax = std::max( Bf,std::max(Rf,Gf));
            mymin = std::min( Bf,std::min(Rf,Gf));
            // RGB -> Mid
            if ((mymax==Rf)&&(mymin==Gf))
            Mid=Bf;
            else if ((mymax==Rf)&&(mymin==Bf))
            Mid=Gf;
            else if ((mymin==Rf)&&(mymax==Gf))
            Mid=Bf;
            else if ((mymin==Rf)&&(mymax==Bf))
            Mid=Gf;
            else if ((mymax==Gf)&&(mymin==Bf))
            Mid=Rf;
            else if ((mymin==Gf)&&(mymax==Bf))
            Mid=Rf;

            // MaxMidMin -> L1re [0,255]
            L1re=(mymax+Mid+mymin)/3.0;

            // MaxMinMidL1re ->  S1re [0,255]
            if ((mymax+mymin)>=(2*Mid)){
            S1re=(3.0*(mymax-L1re))/2.0;
            }
            else {
            S1re=(3.0*(L1re-mymin))/2.0;
            }

            // RGB -> lambda [0,1,2,3,4,5]
            if ((Rf>=Gf)&&(Gf>=Bf))
            lambda=0;
            else if ((Gf>=Rf)&&(Rf>=Bf))
            lambda=1;
            else if ((Gf>=Bf)&&(Bf>=Rf))
            lambda=2;
            else if ((Bf>=Gf)&&(Gf>=Rf))
            lambda=3;
            else if ((Bf>=Rf)&&(Rf>=Gf))
            lambda=4;
            else if ((Rf>=Bf)&&(Bf>=Gf))
            lambda=5;

            // lambda phi k -> k=42 => Hre [0,252]
            //              -> k=60 => Hre [0,360]
            //k=42.000;
            k=60.0;

            // MaxMidMin -> phi [0,1]
            if (S1re==0)
            phi=0;
            else{
            //phi=(1/2)-((mymax+mymin-2*Mid)/(2*S1re));
            //phi=(1/2)-(pow(-1,lambda))*((mymax+mymin-2*Mid)/(2*S1re));
            if ((lambda==0)||(lambda==2)||(lambda==4))
            phi=(0.5)-((mymax+mymin-2*Mid)/(2*S1re));
            else
            phi=0.5+((mymax+mymin-2*Mid)/(2*S1re));
            }

            if (S1re==0)
            Hre=0;
            else {
            //Hre=k*(lambda+0.5+phi);//BMI NOV 2010, erroneous version, with an extra 0.5
            Hre=k*(lambda+phi);//BMI NOV 2010
            }

            // L1reHreS1re -> L1ouHouS1ou
            L1ou=L1re*255.0;
            Hou=((Hre*255.0)/360.0);
            S1ou=S1re*255;
            

            if (L1ou>255) L1ou=255;
            if (L1ou<0) L1ou=0;
            if (Hou>255) Hou=255;
            if (Hou<0) Hou=0;
            if (S1ou>255) S1ou=255;
            if (S1ou<0) S1ou=0;

            h[i] = floor(0.5+Hou);
            l[i] = floor(0.5+L1ou);
            s[i] = floor(0.5+S1ou);
        }
        

        imHlsOut.modified();
        
        return RES_OK;
    }
    

    RES_T HLSToRGB(const Image<RGB> &imHlsIn, Image<RGB> &imRgbOut)
    {
        ASSERT_ALLOCATED(&imHlsIn, &imRgbOut);
        ASSERT_SAME_SIZE(&imHlsIn, &imRgbOut);
        
        Image<UINT8>::lineType h = imHlsIn.getPixels().arrays[0];
        Image<UINT8>::lineType l = imHlsIn.getPixels().arrays[1];
        Image<UINT8>::lineType s = imHlsIn.getPixels().arrays[2];
        
        Image<UINT8>::lineType r = imRgbOut.getPixels().arrays[0];
        Image<UINT8>::lineType g = imRgbOut.getPixels().arrays[1];
        Image<UINT8>::lineType b = imRgbOut.getPixels().arrays[2];
        
      #pragma omp for
        for (size_t i=0;i<imHlsIn.getPixelCount();i++)
        {
            float phi1,phi;
            float lambda1;
            int    lambda;
            float Rou,Gou,Bou;
            float k;

            /* H, L, S are extracted */
            float H = static_cast<float> (h[i]);
            float L = static_cast<float> (l[i]);
            float S = static_cast<float> (s[i]);

            L=((float)(L)/255.0)*(1.0);
            H=((float)H*360.0/255.0);
            S=((float)(S)/255.0)*(1.0);


            //k
            k=60.0;

            // H -> phi1
            lambda1=H/k;

            lambda=floor(lambda1);

            phi=(H/k)-lambda;
            phi1=phi+0.0;
            if(lambda >= 6){// BMI NOV 2010, "no case lambda 6 afterwards". Be careful, phi definition should be before lambda becomes 0.
            lambda = 0;
            }

            // L1reS1re phi -> RouGouBou
            if (phi<=0.5) {
            if (lambda==0){         //Even

            Rou=L+(2.0/3.0)*S;
            Gou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
            Bou=L-(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            }
            else if (lambda==1) {   //Odd
            Gou=L+(2.0/3.0)*S;
            Rou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            Bou=L-S+(2.0/3.0)*S*phi1;
            }
            else if (lambda==2) {
            Gou=L+(2.0/3.0)*S;
            Bou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
            Rou=L-(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            }
            else if (lambda==3) {
            Bou=L+(2.0/3.0)*S;
            Gou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            Rou=L-S+(2.0/3.0)*S*phi1;
            }
            else if (lambda==4) {
            Bou=L+(2.0/3.0)*S;
            Rou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
            Gou=L-(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            }
            else if (lambda==5) {
            Rou=L+(2.0/3.0)*S;
            Bou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            Gou=L-S+(2.0/3.0)*S*phi1;
            }
            }
            else {
            if (lambda==0)  {
                Rou=L+S-(2.0/3.0)*S*phi1;
                Gou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
                Bou=L-(2.0/3.0)*S;
            }
            else if (lambda==1) {
                Gou=L+(1.0/3.0)*S+(2.0/3.0)*S*phi1;
                Rou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
                Bou=L-(2.0/3.0)*S;
            }
            else if (lambda==2){
                Gou=L+S-(2.0/3.0)*S*phi1;
                Bou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
                Rou=L-(2.0/3.0)*S;
            }
            else if (lambda==3) {
            Bou=L+(1.0/3.0)*S+(2.0/3.0)*S*phi1;
            Gou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            Rou=L-(2.0/3.0)*S;
            }
            else if (lambda==4) {
                Bou=L+S-(2.0/3.0)*S*phi1;
                Rou=L-(1.0/3.0)*S+(2.0/3.0)*S*phi1;
                Gou=L-(2.0/3.0)*S;
            }
            else if (lambda==5) {
            Rou=L+(1.0/3.0)*S+(2.0/3.0)*S*phi1;
            Bou=L+(1.0/3.0)*S-(2.0/3.0)*S*phi1;
            Gou=L-(2.0/3.0)*S;
            }
            }


            Rou=floor(0.5+Rou*255);//BMI 2010;
            Gou=floor(0.5+Gou*255);
            Bou=floor(0.5+Bou*255);

            if (Rou>255) Rou=255;
            if (Rou<0) Rou=0;
            if (Gou>255) Gou=255;
            if (Gou<0) Gou=0;
            if (Bou>255) Bou=255;
            if (Bou<0) Bou=0;

            r[i] = Rou;
            g[i] = Gou;
            b[i] = Bou;
        }
        

        imRgbOut.modified();
        
        return RES_OK;
    }
    
    
    
    
    RES_T RGBToLAB(const Image<RGB> &imRgbIn, Image<RGB> &imLabOut)
    {
        ASSERT(RGBToXYZ(imRgbIn, imLabOut)==RES_OK);
        return XYZToLAB(imLabOut, imLabOut);
    }
    
    RES_T LABToRGB(const Image<RGB> &imLabIn, Image<RGB> &imRgbOut)
    {
        ASSERT(LABToXYZ(imLabIn, imRgbOut)==RES_OK);
        return XYZToRGB(imRgbOut, imRgbOut);
    }
    
} // namespace smil

