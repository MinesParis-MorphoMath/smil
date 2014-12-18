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


#include "Core/include/DCore.h"
#include "DMorpho.h"

using namespace smil;

class Test_Dilate_2Points : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        114, 133,  74, 160,  57,  25,  37,
         23,  73,   9, 196, 118,  23, 110,
        154, 248, 165, 159, 210,  47,  58,
        213,  74,   8, 163,   3, 240, 213,
        158,  67,  52, 103, 163, 158,   9,
         85,  36, 124,  12,   7,  56, 253,
        214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      StrElt dse;
      dse.addPoint(0,0);
      dse.addPoint(1,1);
      
      dilate(im1, im2, dse());

      dataType dilateVec[] = {
        114, 133,  74, 160,  57,  25,  37,
          23, 114, 133, 196, 160,  57, 110,
        154, 248, 165, 159, 210, 118,  58,
        213, 154, 248, 165, 159, 240, 213,
        158, 213,  74, 103, 163, 158, 240,
          85, 158, 124,  52, 103, 163, 253,
        214, 148,  36, 200,  53,  10,  58,
      };
      im3 << dilateVec;
      
      
      TEST_ASSERT(im2==im3);      
      
      if (retVal!=RES_OK)
        im2.printSelf(1);
  }
};

class Test_3Points_Hex : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        255,    0,    0,    0,    0,    0,    0,
             0,    0,  255,  255,    0,    0,  255,
        255,    0,  255,   255,  255,  255,    0,
             0,    0,    0,    0,  255,    0,  255,
        255,  255,    0,   255,    0,    0,    0,
             0,    0,   255,   255,    0,    0,  255,
        255,    0,    0,    0,    0,    0,    0,
      };
      
      im1 << vec1;
      
      StrElt se(1,  3,  0,1,2);
     
      erode(im1, im2, se);

      dataType erodeVec[] = {
        0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,  255,
        0,    0,  255,  255,    0,    0,    0,
          0,    0,    0,    0,    0,    0,  255,
        0,    0,    0,    0,    0,    0,    0,
          0,    0,  255,    0,    0,    0,  255,
        0,    0,    0,    0,    0,    0,    0,
      };
      im3 << erodeVec;
      
      TEST_ASSERT(im2==im3);      

      if (retVal!=RES_OK)
        im2.printSelf(1,1);
      
      dilate(im2, im1, se);
      
      dataType dilateVec[] = {
        0,    0,    0,    0,    0,    0,    0,
          0,    0,  255,  255,    0,    0,  255,
        0,    0,  255,  255,  255,    0,    0,
          0,    0,    0,    0,    0,    0,  255,
        0,    0,    0,  255,    0,    0,    0,
          0,    0,  255,  255,    0,    0,  255,
        0,    0,    0,    0,    0,    0,    0,
      };
      im3 << dilateVec;
      
      TEST_ASSERT(im1==im3);      
      
      if (retVal!=RES_OK)
        im1.printSelf(1, true);
  }
};

class Test_Dilate_Vert : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        114, 133,  74, 160,  57,  25,  37,
         23,  73,   9, 196, 118,  23, 110,
        154, 248, 165, 159, 210,  47,  58,
        213,  74,   8, 163,   3, 240, 213,
        158,  67,  52, 103, 163, 158,   9,
         85,  36, 124,  12,   7,  56, 253,
        214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dilate(im1, im2, VertSE());

      dataType dilateVec[] = {
        114, 133,  74, 196, 118,  25, 110,
        154, 248, 165, 196, 210,  47, 110,
        213, 248, 165, 196, 210, 240, 213,
        213, 248, 165, 163, 210, 240, 213,
        213,  74, 124, 163, 163, 240, 253,
        214, 148, 124, 200, 163, 158, 253,
        214, 148, 124, 200,  53,  56, 253,
      };
      im3 << dilateVec;
      
      
      TEST_ASSERT(im2==im3);      
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        diff(im2, im3, im3);
        im3.printSelf(1);
      }
  }
};

class Test_Dilate_Cross : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,9);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        114, 133,  74, 160,  57,  25,  37,
         23,  73,   9, 196, 118,  23, 110,
        154, 248, 165, 159, 210,  47,  58,
        213,  74,   8, 163,   3, 240, 213,
        158,  67,  52, 103, 163, 158,   9,
         85,  36, 124,  12,   7,  56, 253,
        214, 148,  20, 200,  53,  10,  58,
        214, 148,  20, 200,  53,  10,  58,
        213,  74,   8, 163,   3, 240, 213,
      };
      
      im1 << vec1;
      
      dilate(im1, im1, CrossSE());

      dataType dilateVec[] = {
        133, 133, 160, 196, 160,  57, 110,
        154, 248, 196, 196, 210, 118, 110,
        248, 248, 248, 210, 210, 240, 213,
        213, 248, 165, 163, 240, 240, 240,
        213, 158, 124, 163, 163, 240, 253,
        214, 148, 124, 200, 163, 253, 253,
        214, 214, 200, 200, 200,  58, 253,
        214, 214, 200, 200, 200, 240, 213,
        214, 213, 163, 200, 240, 240, 240,
      };
      im3 << dilateVec;
      
      
      TEST_ASSERT(im1==im3);      
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        diff(im2, im3, im3);
        im3.printSelf(1);
      }
  }
};

class Test_Dilate_Hex : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        114, 133,  74, 160,  57,  25,  37,
         23,  73,   9, 196, 118,  23, 110,
        154, 248, 165, 159, 210,  47,  58,
        213,  74,   8, 163,   3, 240, 213,
        158,  67,  52, 103, 163, 158,   9,
         85,  36, 124,  12,   7,  56, 253,
        214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateHexVec[] = {
        133, 133, 160, 196, 196, 118, 110, 
        248, 248, 196, 210, 210, 118, 110, 
        248, 248, 248, 210, 210, 240, 240, 
        248, 248, 165, 210, 240, 240, 240, 
        213, 213, 124, 163, 163, 240, 253, 
        214, 148, 200, 200, 163, 253, 253, 
        214, 214, 200, 200, 200, 58, 253
      };
      im3 << dilateHexVec;
      
      // The specialized way
      dilate(im1, im2, hSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = hSE().points;
      se.odd = true;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      
      
      // With an homothetic SE
      dilate(im1, im3, hSE(3));
      dilate(im1, im2, hSE().homothety(3));
      TEST_ASSERT(im2==im3);
      
  }
};


class Test_Dilate_Squ : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        114, 133,  74, 160,  57,  25,  37,
         23,  73,   9, 196, 118,  23, 110,
        154, 248, 165, 159, 210,  47,  58,
        213,  74,   8, 163,   3, 240, 213,
        158,  67,  52, 103, 163, 158,   9,
         85,  36, 124,  12,   7,  56, 253,
        214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateSquVec[] = {
        133, 133, 196, 196, 196, 118, 110, 
        248, 248, 248, 210, 210, 210, 110, 
        248, 248, 248, 210, 240, 240, 240, 
        248, 248, 248, 210, 240, 240, 240, 
        213, 213, 163, 163, 240, 253, 253, 
        214, 214, 200, 200, 200, 253, 253, 
        214, 214, 200, 200, 200, 253, 253
      };
      im3 << dilateSquVec;

      // The specialized way
      dilate(im1, im2, sSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = sSE().points;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      

      
      dataType dilateSquVec2[] = {
        248, 248, 248, 248, 210, 210, 210,
        248, 248, 248, 248, 240, 240, 240,
        248, 248, 248, 248, 240, 240, 240,
        248, 248, 248, 248, 253, 253, 253,
        248, 248, 248, 248, 253, 253, 253,
        214, 214, 214, 240, 253, 253, 253,
        214, 214, 214, 200, 253, 253, 253,
      };
      im3 << dilateSquVec2;
      
      // The specialized way
      dilate(im1, im2, sSE(2));
      TEST_ASSERT(im2==im3);      
      
      // With an homothetic SE
      dilate(im1, im2, sSE().homothety(2));
      TEST_ASSERT(im2==im3);
//       im2.printSelf(1);
  }
};


class Test_Dilate_3D : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(5,5,5);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
        207, 170, 100, 107, 141,
        230, 233,  99, 245, 115,
          71, 112, 121, 177, 141,
        155, 153, 109,  85, 134,
        147, 122, 106,  72, 173,

        204, 221, 116,  19,  91,
        231, 227,   6,  77,  80,
        148,  53,  58, 248,  43,
        174,  64, 156,  41, 241,
          42, 197, 139, 152,  27,

          19,  45, 149, 125, 118,
          67,  75,  84, 183,  95,
        176, 160,  67, 183, 238,
        148,  76,  36, 206,  69,
          80, 125, 134, 236, 167,

        120,  24, 109,   5, 176,
        136,  24, 222,  31, 149,
          85,  99, 224, 170,  27,
          65,  91, 188, 132,  20,
        172,  25,  96, 208, 232,

        166,  33, 103,  45,  15,
          15, 166,  39, 125, 171,
        216,   6, 195, 184,  37,
          90,  14, 136,  60, 184,
        164, 125,  21,  98,   3,
      };
      
      im1 << vec1;
      
      dataType dilateVec[] = {
        233, 233, 245, 245, 245,
        233, 233, 248, 248, 248,
        233, 233, 248, 248, 248,
        197, 197, 248, 248, 248,
        197, 197, 197, 241, 241,

        233, 233, 245, 245, 245,
        233, 233, 248, 248, 248,
        233, 233, 248, 248, 248,
        197, 197, 248, 248, 248,
        197, 197, 236, 241, 241,

        231, 231, 227, 222, 183,
        231, 231, 248, 248, 248,
        231, 231, 248, 248, 248,
        197, 224, 248, 248, 248,
        197, 197, 236, 241, 241,

        166, 222, 222, 222, 183,
        216, 224, 224, 238, 238,
        216, 224, 224, 238, 238,
        216, 224, 236, 238, 238,
        172, 188, 236, 236, 236,

        166, 222, 222, 222, 176,
        216, 224, 224, 224, 184,
        216, 224, 224, 224, 184,
        216, 224, 224, 232, 232,
        172, 188, 208, 232, 232,
      };
      im3 << dilateVec;

      // The specialized way
      dilate(im1, im2, CubeSE());
      TEST_ASSERT(im2==im3);      
  }
};

class Test_Dilate_Rhombicuboctahedron : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7,7);
      imType im2(im1);
      imType im3(im1);
   
      dataType vec1[] = {
          90,  49, 153, 203,   9,  77,  25,
         223,  89,   3,  71, 133,  57, 196,
         238, 226,  92,  22,   5, 226,  24,
          47, 252,  10,  17,  80, 240,  73,
         112, 225,  19, 202,  19, 172, 150,
          28, 249, 175, 252,  84, 179,  68,
         218, 236,  10, 202, 207, 102, 224,

         213,  74, 248,   5,  72,   4,  23,
         152, 244,  96,   9, 215, 116, 211,
         234,  34, 106,   8,  28,  27,   5,
         113, 207,  73,  77, 188,  84,  24,
         140, 186, 248,  99,   5, 242, 105,
          77, 246, 128, 229, 235, 224, 239,
         195,  86, 195, 175, 120,  47, 184,

         149,  75, 189,   7,  27,   8,  85,
         216,  92, 109, 102,  24, 102, 201,
          29,  90,  51, 107,  81, 179,  81,
          62, 149,  65,   3, 235,   6, 178,
         101,  54, 108, 250, 129,  42,   3,
         157,  51,  88, 119, 144, 197, 221,
         168,  45, 167, 198, 135, 218,  50,

         217, 143, 131,  25,  38, 197,  28,
          19, 204, 207, 120,   3,  60, 115,
         133, 103, 118,  35, 155, 206, 154,
          44, 149, 121, 213, 195,  33, 156,
          75, 252, 206,  38, 141,  83,  63,
         179,  25,  92, 199, 229,  45,  64,
         232, 105, 179, 110, 209,  43, 146,

         109, 250,  46, 154, 144, 167, 112,
          84, 201,  13, 160, 198, 219, 199,
          85,  47,   8,  10,  73, 100, 209,
          48, 146,  18,  25, 251, 198, 136,
         206, 241,  28,  61, 236,  74, 215,
         126, 242,  72, 210, 188,  86, 116,
         132,  51,  60, 217,  99,  68, 227,

         172, 169, 182, 220,  60, 200, 246,
          57, 144, 127,   8, 130, 156,  70,
         112, 230,  30, 239, 217, 103, 194,
         151, 189,  56,  28, 240, 116, 246,
          84, 185, 219,   2,  99, 146, 222,
         159,  92, 213, 217, 236,  86, 226,
         112, 242,  41, 225, 218,  71, 209,

         181, 175, 149,  78, 109, 205, 106,
          95,  66,  97, 180, 251,  61, 182,
          96, 208, 149,   1,  45, 108, 218,
          27, 195, 189, 139, 183, 231, 110,
         146,  48,  64,  73, 223, 213, 151,
          78, 163,   3, 173, 230, 101,  98,
         226, 163,  25,  68, 116, 175,  69
      };
      
      im1 << vec1;
      
      dataType dilateRcoVec[] = {
          252, 252, 252, 248, 248, 248, 240,
          252, 252, 252, 252, 248, 248, 242,
          252, 252, 252, 252, 252, 242, 242,
          252, 252, 252, 252, 252, 252, 242,
          252, 252, 252, 252, 252, 252, 252,
          252, 252, 252, 252, 252, 252, 252,
          252, 252, 252, 252, 252, 252, 252,

          252, 252, 252, 248, 248, 248, 240,
          252, 252, 252, 252, 250, 248, 242,
          252, 252, 252, 252, 252, 251, 242,
          252, 252, 252, 252, 252, 252, 250,
          252, 252, 252, 252, 252, 252, 252,
          252, 252, 252, 252, 252, 252, 252,
          252, 252, 252, 252, 252, 252, 252,

          250, 250, 250, 250, 248, 248, 246,
          252, 252, 252, 251, 251, 251, 246,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 252, 251,
          252, 252, 252, 252, 252, 252, 250,
          252, 252, 252, 252, 252, 252, 242,

          250, 250, 250, 251, 251, 251, 246,
          252, 252, 252, 251, 251, 251, 251,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 251, 242,

          250, 250, 251, 251, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 251,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 252, 251, 251,
          252, 252, 252, 252, 251, 251, 246,

          250, 251, 251, 251, 251, 251, 251,
          250, 251, 251, 251, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 251,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 252, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 246,

          250, 251, 251, 251, 251, 251, 251,
          250, 251, 251, 251, 251, 251, 251,
          250, 251, 251, 251, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 251,
          252, 252, 252, 251, 251, 251, 246,
          242, 242, 242, 242, 242, 246, 246,
      };
      
      im3 << dilateRcoVec;

      // The specialized way
      dilate(im1, im2, rcoSE(3));
      TEST_ASSERT(im2==im3);      
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Dilate_2Points);
      ADD_TEST(ts, Test_3Points_Hex);
      ADD_TEST(ts, Test_Dilate_Vert);
      ADD_TEST(ts, Test_Dilate_Cross);
      ADD_TEST(ts, Test_Dilate_Hex);
      ADD_TEST(ts, Test_Dilate_Squ);
      ADD_TEST(ts, Test_Dilate_3D);
      ADD_TEST(ts, Test_Dilate_Rhombicuboctahedron);
      
//       UINT BENCH_NRUNS = 5E3;
//       Image_UINT8 im1(1024, 1024), im2(im1);
//       BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
//       BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());
// cout << endl;
//       tc(im1, im2, sSE());
      return ts.run();
  
}

