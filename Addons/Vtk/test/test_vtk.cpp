
#include "DVtkInterface.hpp"
#include "Core/include/DTest.h"


using namespace smil;

class Test_Import : public TestCase
{
    virtual void run()
    {
        vtkImageData *imData = vtkImageData::New();
        imData->SetExtent(0, 49, 0, 49, 0, 49);
        imData->SetScalarTypeToUnsignedChar();
        imData->SetNumberOfScalarComponents(1); // image holds one value intensities
        imData->AllocateScalars(); // allocate
        
        VtkInt<UINT8> sIm(imData);
        TEST_ASSERT(sIm.getWidth()==50);
        TEST_ASSERT(sIm.getHeight()==50);
        TEST_ASSERT(sIm.getDepth()==50);
        
        imData->Delete();
    }
};

int main()
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Import);
      
      
      return ts.run();
  
}

