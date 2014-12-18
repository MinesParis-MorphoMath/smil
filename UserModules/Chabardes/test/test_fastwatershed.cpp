#include "DWatershed.hpp"

using namespace smil;

class Test_FastWatershed:public TestCase
{
    virtual void run ()
    {
        UINT8 vecIn[] = {
1,  1,  1,  2,  5,  10, 5,  5,  4,  4,
1,  1,  1,  2,  5,  10, 5,  5,  4,  4,
3,  4,  4,  8,  6,  11, 6,  5,  11, 9,
12, 8,  9,  8,  11, 6,  7,  8,  7,  12,
5,  12, 4,  12, 9,  10, 7,  8,  12, 12,
3,  2,  7,  6,  12, 3,  7,  5,  5,  12,
1,  3,  4,  9,  8,  13, 5,  3,  9,  11,
5,  5,  5,  6,  13, 10, 9,  0,  5,  11,
3,  5,  2,  7,  6,  9,  8,  13, 9,  12,
5,  5,  7,  5,  6,  8,  12, 9,  13, 10
        };
/*
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
*/
        Image_UINT8 imIn (10, 10);
    Image_UINT8 imMarker (imIn);
    Image_UINT8 imWatershed (imIn);
    Image_UINT8 imBasins (imIn);

        StrElt se = cSE ();

        imIn << vecIn;

    fastWatershed (imIn, imMarker, imWatershed, imBasins, se);
//    fastMinima (imIn, imWatershed, se);

//        TEST_ASSERT (label == truth);
    }
};

class Test_Watershed:public TestCase
{
    virtual void run ()
    {
        UINT8 vecIn[] = {
1,  1,  1,  2,  5,  10, 5,  5,  4,  4,
1,  1,  1,  2,  5,  10, 5,  5,  4,  4,
3,  4,  4,  8,  6,  11, 6,  5,  11, 9,
12, 8,  9,  8,  11, 6,  7,  8,  7,  12,
5,  12, 4,  12, 9,  10, 7,  8,  12, 12,
3,  2,  7,  6,  12, 3,  7,  5,  5,  12,
1,  3,  4,  9,  8,  13, 5,  3,  9,  11,
5,  5,  5,  6,  13, 10, 9,  0,  5,  11,
3,  5,  2,  7,  6,  9,  8,  13, 9,  12,
5,  5,  7,  5,  6,  8,  12, 9,  13, 10
        };

        Image_UINT8 imIn (10, 10);
    Image_UINT8 imMarker (imIn);
    Image_UINT8 imWatershed (imIn);
    Image_UINT8 imBasins (imIn);

        StrElt se = cSE ();

        imIn << vecIn;

    fastMinima (imIn, imMarker, se) ;

    fastWatershed (imIn, imMarker, imWatershed, imBasins, se);

//        TEST_ASSERT (label == truth);
    }
};

int
main (int argc, char *argv[])
{
    TestSuite ts;

    ADD_TEST (ts, Test_Watershed);
    ADD_TEST (ts, Test_FastWatershed);

    return ts.run ();

}
