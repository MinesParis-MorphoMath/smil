

#include "Core/include/DCore.h"
#include "Addons/FFT/include/DFFT.hpp"


using namespace smil;


class Test_Correlation : public TestCase
{
    virtual void run()
    {
        UINT8 vec1[35] = 
        {
          0, 0, 0, 1, 0, 1, 0,
          0, 0, 1, 0, 0, 16, 0,
          0, 1, 0, 0, 0, 1, 0,
          1, 1, 128, 1, 0, 1, 0,
          0, 0, 0, 0, 0, 1, 0,
        };
        
        UINT8 vec2[35] = 
        {
          1, 0, 0, 16, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0,
          128, 1, 0, 1, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
        };
        
        Image<UINT8> im1(7, 5);
        Image<UINT8> im2(7, 5);
        Image<UINT8> im3(7, 5);
        
        im1 << vec1;
        im2 << vec2;
        
        correlation(im1, im2, im3);
        
        UINT8 vecTruth[35] = 
        {
            128, 128, 127, 127, 127, 127, 127,
            128, 128, 255, 127, 127, 128, 127,
            128, 127, 128, 127, 127, 128, 128,
            128, 128, 127, 128, 127, 128, 127,
            128, 127, 128, 127, 127, 128, 127,
        };
        Image<UINT8> imTruth(7, 5);
        imTruth << vecTruth;
        
        TEST_ASSERT(im3==imTruth);
    }
};


int main(int, char *[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Correlation);

      return ts.run();
  
}

