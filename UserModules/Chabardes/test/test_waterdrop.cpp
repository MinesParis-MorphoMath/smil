#include "DWatershed.h"

using namespace smil;

class Test_Waterdrop : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	2, 2, 2, 2, 2, 2,
	7, 7, 7, 7, 7, 7,
	2, 7, 5, 6, 2, 2,
	2, 6, 5, 6, 2, 2,
	2, 2, 6, 4, 3, 2,
	2, 2, 3, 4, 2, 2,
	2, 2, 2, 2, 4, 2
      };
      
      UINT8 vecLbl[] = { 
	1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 3, 3,
	2, 0, 0, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };
       
      UINT8 vecTruth[] = {
	1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0,
	2, 0, 2, 0, 3, 3,
	2, 0, 2, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };

      Image_UINT8 in(6,7);
      Image_UINT8 label(in);
      Image_UINT8 arrow(in);
      Image_UINT8 truth(in);

      StrElt se = hSE();
      
      in << vecIn;
      label << vecLbl;
      arrowSteepest (in, arrow, se, UINT8(255)) ;
      truth << vecTruth;

      //waterdropFunc func ;
      //func (in, arrow, 14, label, se) ;
      
      TEST_ASSERT (label == truth) ;
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Waterdrop);
      
      return ts.run();
      
}


