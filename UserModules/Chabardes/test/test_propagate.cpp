#include "DWatershed.hpp"

using namespace smil;

class Test_Propagate: public TestCase
{
    virtual void run ()
    {
        UINT8 vecI[] = 
        {
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

        UINT8 vecO[] = 
        {
0,  0,  0,  0,  0,  0,  5,  5,  4,  4,
0,  0,  0,  0,  0,  0,  5,  5,  4,  4,
0,  0,  0,  0,  0,  0,  6,  5,  11, 9,
0,  0,  0,  0,  0,  6,  7,  8,  7,  12,
5,  0,  4,  0,  9,  10, 7,  8,  12, 12,
3,  2,  7,  6,  12, 3,  7,  5,  5,  12,
1,  3,  4,  9,  8,  13, 5,  3,  9,  11,
5,  5,  5,  6,  13, 10, 9,  0,  5,  11,
3,  5,  2,  7,  6,  9,  8,  13, 9,  12,
5,  5,  7,  5,  6,  8,  12, 9,  13, 10
        };


        Image_UINT8 imIn (10, 10);
        Image_UINT8 arrowing (imIn);
        Image_UINT8 imOut (imIn);

        StrElt se = cSE().noCenter();

        imIn << vecI;
        imOut << vecO;

        arrow (imIn, "<=", arrowing, se.noCenter (), UINT8(0));
        bredthFirst (arrowing, imIn, se, 0, UINT8(0));

        TEST_ASSERT (imIn == imOut);
    }
};

int main (int argc, char *argv[])
{
    TestSuite ts;
    ADD_TEST (ts, Test_Propagate);
    return ts.run ();
}
