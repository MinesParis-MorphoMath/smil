#include <morphee/common/include/commonError.hpp>
#include <morphee/imageIO/include/morpheeImageIO.hpp>
#include <morphee/image/include/imageArithmetic.hpp>

#include <morphee/filters/include/private/filtersHarris_T.hpp>

// This test program requires Boost::Test
#include <boost/test/unit_test.hpp>

#include <limits>
using namespace boost::unit_test_framework;

extern test_suite *filtersGlobalTest;

using namespace morphee;
using namespace morphee::filters;

class HarrisTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&testHarris));
  }

  static void testHarris()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("HarrisTests::testHarris:");
    Image<F_DOUBLE> im(5, 5);
    im.allocateImage();

    std::string s = "0. 0. 0. 0. 0.\
							 0. 9. 9. 9. 0.\
							 0. 9. 9. 9. 0.\
							 0. 9. 9. 9. 0.\
							 0. 0. 0. 0. 0.";
    std::stringstream sst(s);
    sst >> im;

    Image<F_DOUBLE> imOut(5, 5);
    imOut.allocateImage();

    t_ImSetConstant(imOut, 3.14159);
    unsigned int sz = 1;
    BOOST_CHECK(t_ImHarrisOperator(im, sz, imOut) == RES_OK);

    BOOST_MESSAGE("imOut : " << imOut);
  }
};

void test_harris()
{
  HarrisTests::addTests();
}
