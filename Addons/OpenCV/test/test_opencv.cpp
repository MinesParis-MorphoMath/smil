
#include "DOpenCVInterface.hpp"
#include "Core/include/DTest.h"


using namespace smil;
using namespace cv;

class Test_CvToSmil : public TestCase
{
    virtual void run()
    {
        IplImage* img = cvCreateImage(cvSize(5,5),IPL_DEPTH_8U,1);
        cvZero(img);
        for (int i=0;i<25;i++)
          img->imageData[i] = i;
        OpenCVInt<UINT8> cvInt(img);
        for (int i=0;i<25;i++)
          TEST_ASSERT(cvInt[i]==i);
        cvReleaseImage(&img);
    }
};

class Test_SmilToCv : public TestCase
{
    virtual void run()
    {
        Image<UINT8> sIm(5,5);
        for (int i=0;i<25;i++)
          sIm[i] = i;
        Mat cvMat = toMatImage(sIm);
        for (int i=0;i<25;i++)
          TEST_ASSERT(cvMat.data[i]==i);
    }
};



int main(int /*argc*/, char */*argv*/[])
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_CvToSmil);
      ADD_TEST(ts, Test_SmilToCv);
      
      return ts.run();
  
}

