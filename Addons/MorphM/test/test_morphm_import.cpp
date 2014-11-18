
#include "DMorphMImage.hpp"
#include "Core/include/DCore.h"


using namespace smil;

class Test_MorphM_Import : public TestCase
{
    virtual void run()
    {
//         morphee::Image<UINT8> *mIm = new morphee::Image<UINT8>(256,256);
//         
//         MorphmInt<UINT8> mInt(*mIm);
//         mInt.printSelf();
//         
//         delete mIm;
//         mIm->pr
    }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_MorphM_Import);
      
      return ts.run();
  
}

