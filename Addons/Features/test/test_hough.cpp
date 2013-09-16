

#include "DTest.h"
#include "DHoughTransform.hpp"


using namespace smil;


class Test_Hough_Lines : public TestCase
{
    virtual void run()
    {
	UINT8 vec1[35] = 
	{
	  0, 0, 0, 1, 0, 1, 0,
	  0, 0, 1, 0, 0, 1, 0,
	  0, 1, 0, 0, 0, 1, 0,
	  1, 1, 1, 1, 0, 1, 0,
	  0, 0, 0, 0, 0, 1, 0,
	};
	
	Image<UINT8> im1(7,5);
	Image<UINT8> im2;
	
	im1 << vec1;
	
	houghLines(im1, 1, 10, im2);
	
	TEST_ASSERT(im2.getPixel(0,49)==5);
	TEST_ASSERT(im2.getPixel(90,29)==5);
	TEST_ASSERT(im2.getPixel(45,21)==4);
    }
};

class Test_Hough_Circles : public TestCase
{
    virtual void run()
    {
	UINT8 vec1[35] = 
	{
	  0, 1, 0, 0, 0, 0, 0,
	  1, 0, 1, 0, 1, 1, 0,
	  0, 1, 0, 1, 0, 0, 1,
	  0, 0, 0, 1, 0, 0, 1,
	  0, 0, 0, 0, 1, 1, 0,
	};
	
	Image<UINT8> im1(7,5);
	Image<UINT8> im2(im1);
	
	im1 << vec1;
	
	houghCircles(im1, 10, im2);
	
	TEST_ASSERT(im2.getPixel(10,10,10)==4);
	TEST_ASSERT(im2.getPixel(45,25,15)==8);
    }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Hough_Lines);
      ADD_TEST(ts, Test_Hough_Circles);

      return ts.run();
  
}

