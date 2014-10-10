#include "DWatershed.h"

using namespace smil;

class Test_Waterdrop:public TestCase
{
    virtual void run ()
    {
	UINT8 vecIn[] = {
	    4, 4, 4, 4, 4, 4,
	    7, 7, 7, 7, 7, 7,
	    2, 7, 5, 6, 2, 2,
	    2, 6, 5, 6, 2, 2,
	    2, 2, 6, 4, 1, 2,
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

	Image_UINT8 imIn (6, 7);
    Image_UINT8 imMinima (imIn);
	Image_UINT8 label (imIn);
	Image_UINT8 truth (imIn);

	StrElt se = cSE ();

	imIn << vecIn;
	label << vecLbl;
	truth << vecTruth;

    fastMinima (imIn, imMinima, se);

    imMinima.printSelf (1);

//	TEST_ASSERT (label == truth);
    }
};


int
main (int argc, char *argv[])
{
    TestSuite ts;

    ADD_TEST (ts, Test_Waterdrop);

    return ts.run ();

}
