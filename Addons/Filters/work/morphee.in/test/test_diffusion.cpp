#include <morphee/common/include/commonError.hpp>
#include <morphee/imageIO/include/morpheeImageIO.hpp>
#include <morphee/image/include/imageArithmetic.hpp>
#include <morphee/image/include/private/image_T.hpp>

#include <morphee/filters/include/private/filtersDiffusion_T.hpp>

// This test program requires Boost::Test
#include <boost/test/unit_test.hpp>
using namespace boost::unit_test_framework;

extern test_suite *filtersGlobalTest;
using namespace morphee;

class diffusionTest
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&gradientTest));
    filtersGlobalTest->add(BOOST_TEST_CASE(&stupidTest));
  }

  static void stupidTest()
  {
    s_logCleaner cleaner;
    // test de fonctionnement
    MORPHEE_RESET_STACK(); //<--- Brutal ca non

    Image<F_SIMPLE> im(6, 5), imOut(6, 5);
    im.allocateImage();
    imOut.allocateImage();

    std::string s = "\
			1 1 1 1 1 1\
			0 2 2 2 2 2\
			0 1 3 2 5 1\
			1 1 2 1 3 1\
			0 0 0 0 1 0";
    std::istringstream st(s);
    st >> im;

    RES_C res = morphee::filters::t_HeatDiffusion(im, 10, 0.1f, imOut);
    BOOST_CHECK(res == RES_OK);
    if (res != RES_OK) {
      MORPHEE_UNWIND_STACK();
      MORPHEE_RESET_STACK();
    }

    res = morphee::filters::t_PeronaMalikDiffusion(im, 10, 0.1f, 10, imOut);
    BOOST_CHECK(res == RES_OK);
    if (res != RES_OK) {
      MORPHEE_UNWIND_STACK();
      MORPHEE_RESET_STACK();
    }

    res = morphee::filters::t_WeickertDiffusion(im, 10, 0.1f, 10, 1, 1, imOut);
    BOOST_CHECK(res == RES_OK);
    if (res != RES_OK) {
      MORPHEE_UNWIND_STACK();
      MORPHEE_RESET_STACK();
    }
  }

  static void gradientTest()
  {
    s_logCleaner cleaner;
    Image<F_SIMPLE> im(6, 5), imGradX(6, 5), imGradY(6, 5);
    // Image<pixel_3<F_SIMPLE> > imOut(6,5);
    im.allocateImage();
    imGradX.allocateImage();
    imGradY.allocateImage();
    // imOut.allocateImage();

    std::string s = "\
			1 1 1 1 1 1\
			0 2 2 2 2 2\
			0 1 3 2 5 1\
			1 1 2 1 3 1\
			0 0 0 0 1 0";
    std::istringstream st(s);
    st >> im;

    std::string str_gradX = /*"\
      0 0 0 0 0 0\
      0 0 0 0 0 0\
      0 0 0.5 1 -0.5 0\
      0 0 0 0 0 0\
      0 0 0 0 0 0";*/
        "\
			0 0 0 0 0 0\
			0 0 0 0 0 0\
			0 0 0.5 1 0 0\
			0 0 0 0.5 0 0\
			0 0 0 0 0 0";
    // The old value -0.5 should not be computed since the point is out of the
    // active window minus 1 in each direction
    std::istringstream ss_gradX(str_gradX);
    ss_gradX >> imGradX;

    std::string str_gradY = "\
			0 0 0 0 0 0\
			0 0 0 0 0 0\
			0 -0.5 0 -0.5 0.5 -0.5\
			0 0 0 0 0 0\
			0 0 0 0 0 0";
    /*"\
      0 0 0 0 0 0\
      -0.5 0 1 0.5 2 0\
      0.5 -0.5 0 -0.5 0.5 -0.5\
      0 -0.5 -1.5 -1 -2 -0.5\
      0 0 0 0 0 0";*/
    // What should we do ? should we stay inside the window needed to compute
    // ALL directions of the gradient or should the window be different for each
    // direction ?
    std::istringstream ss_gradY(str_gradY);
    ss_gradY >> imGradY;

    im.setActiveWindow(1, 1, 0, 4, 3, 1);
    // imGradX.setActiveWindow(1,1,0,4,3,1);
    // imGradY.setActiveWindow(1,1,0,4,3,1);
    imGradX.setActiveWindow(im.ActiveWindow());
    imGradY.setActiveWindow(im.ActiveWindow());

    // Do not set the window of ImOut yet since it should be done inside
    // t_Gradient

    std::vector<Image<F_SIMPLE> *> vect_out;
    // BOOST_CHECK(t_ImSetConstant(imOut, pixel_3<F_SIMPLE>(0,0,0)) == RES_OK);

    RES_C res = morphee::filters::t_Gradient(im, vect_out);
    BOOST_CHECK(res == RES_OK);
    BOOST_REQUIRE(vect_out.size() == 2);

    BOOST_CHECK(vect_out[0]->getActiveWindow() == imGradX.getActiveWindow());
    BOOST_CHECK(vect_out[1]->getActiveWindow() == imGradY.getActiveWindow());

    // Image<pixel_3<F_SIMPLE> >::const_iterator	itgrad, itgradEnd;
    ImageIteratorInterface *itgrad1, *itgradEnd1;
    ImageIteratorInterface *itgrad2, *itgradEnd2;
    Image<F_SIMPLE>::const_iterator itgradX, itgradY;

    for (itgrad1 = vect_out[0]->begin_it(), itgradEnd1 = vect_out[0]->end_it(),
        itgrad2 = vect_out[1]->begin_it(), itgradEnd2 = vect_out[1]->end_it(),
        itgradX = imGradX.begin(), itgradY = imGradY.begin();
         itgrad1->isDifferent(itgradEnd1) && itgrad2->isDifferent(itgradEnd2) &&
         itgradX != imGradX.end() && itgradY != imGradY.end();
         itgrad1->next(), itgrad2->next(), ++itgradX, ++itgradY) {
      F_SIMPLE val = itgrad1->getPixel();
      BOOST_CHECK_MESSAGE(val == *itgradX,
                          "Grad x :" << itgrad1->getPosition() << val
                                     << " != " << *itgradX << "(expected)");

      val = itgrad2->getPixel();
      BOOST_CHECK_MESSAGE(val == *itgradY,
                          "Grad y :" << itgrad2->getPosition() << val
                                     << " != " << *itgradY << "(expected)");
      // BOOST_CHECK_MESSAGE((*itgrad).channel2 == *itgradY , (*itgrad).channel2
      // << " != " << *itgradY << "(expected)");
    }

    delete itgrad1, itgradEnd1, itgrad2, itgradEnd2;
    for (int i = 0; i < 2; i++) {
      delete vect_out[i];
    }
  }
};

void test_Diffusion()
{
  diffusionTest::addTests();
}
