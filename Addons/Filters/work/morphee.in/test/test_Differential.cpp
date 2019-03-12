#include <morphee/common/include/commonError.hpp>
#include <morphee/imageIO/include/morpheeImageIO.hpp>
#include <morphee/image/include/imageArithmetic.hpp>

#include <morphee/filters/include/private/filtersDifferential_T.hpp>

// This test program requires Boost::Test
#include <boost/test/unit_test.hpp>
using namespace boost::unit_test_framework;

extern test_suite *filtersGlobalTest;

using namespace morphee;
class DifferentialTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&testLaplacian));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradientX));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradientY));
  }

  static void testLaplacian()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("DifferentialTests::testLaplacian");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "12 12 18 12 12 \
				 12 -15  -9 -15 12\
				 18  -9  0  -9 18\
				 12 -15  -9 -15 12\
				 12 12 18 12 12";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImLaplacianFilter(im, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK_MESSAGE(*itIn == *itOut,
                          *itIn << " != " << *itOut << " (expected) ");
  }

  static void testGradientX()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("DifferentialTests::testGradientX");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 4 4 4 0\
				  0 4 4 4 0\
				  0 4 4 4 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 0 0 0 0\
				 0 2 0 -2 0\
				 0 2 0 -2 0\
				 0 2 0 -2 0\
				 0 0 0 0 0";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImDifferentialGradientX(im, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK_MESSAGE(*itIn == *itOut,
                          *itOut << " != " << *itIn << " (expected) ");
  }
  static void testGradientY()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("DifferentialTests::testGradientY");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 4 4 4 0\
				  0 4 4 4 0\
				  0 4 4 4 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 0 0 0 0\
				 0 2 2 2 0\
				 0 0 0 0 0\
				 0 -2 -2 -2 0\
				 0 0 0 0 0";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImDifferentialGradientY(im, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK_MESSAGE(*itIn == *itOut,
                          *itOut << " != " << *itIn << " (expected) ");
  }
}; // class  DifferentialTests

void test_Differential()
{
  DifferentialTests::addTests();
}
