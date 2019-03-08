#include <morphee/common/include/commonError.hpp>
#include <morphee/imageIO/include/morpheeImageIO.hpp>
#include <morphee/image/include/imageArithmetic.hpp>

#include <morphee/filters/include/private/filtersConvolve_T.hpp>
#include <morphee/filters/include/private/filtersGaussian_T.hpp>
#include <morphee/filters/include/private/filtersGaussianRecursive_T.hpp>

// This test program requires Boost::Test
#include <boost/test/unit_test.hpp>

#include <limits>
using namespace boost::unit_test_framework;

extern test_suite *filtersGlobalTest;

using namespace morphee;
class BasicConvolutionTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&testIdentity));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testShift));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradient));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradient1b));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradient2));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradient2b));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGradient3));
  }

  static void testIdentity()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testIdentity");
    morphee::Image<INT32> im(3, 3), imOut(3, 3), imKernel(3, 3);
    im.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s = "1 2 3 \
				 4 5 6 \
				 7 8 9";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 0 0 \
				 0 1 0 \
				 0 0 0";
    std::istringstream st2(s2);
    st2 >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeClip);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im);
    filters::t_ImConvolve(im, imSE, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im.begin(), itEnd = im.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }
  static void testShift()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testIdentity");
    morphee::Image<INT32> im(3, 3), im2(3, 3), imOut(3, 3), imKernel(3, 3);
    im.allocateImage();
    im2.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s = "1 2 3 \
				 4 5 6 \
				 7 8 9";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 1 2 \
				 0 4 5 \
				 0 7 8";
    std::istringstream st2(s2);
    st2 >> im2;

    std::string sk = "0 0 0 \
				 1 0 0 \
				 0 0 0";
    std::istringstream stk(sk);
    stk >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeClip);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImConvolve(im, imSE, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }

  static void testGradient()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testGradient");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5), imKernel(3, 3);
    im.allocateImage();
    im2.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 -3 -3 -3 0 \
				 -3 6  3  6 -3\
				 -3 3  0  3 -3\
				 -3 6  3  6 -3\
				 0 -3 -3 -3 0";
    std::istringstream st2(s2);
    st2 >> im2;

    std::string sk = "0 -1 0 \
				 -1 4 -1 \
				 0 -1 0";
    std::istringstream stk(sk);
    stk >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeClip);

    t_ImBorderSetConstant(im, (INT32) 200);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImConvolve(im, imSE, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }
  // Avec border_mirror
  static void testGradient1b()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testGradient1b");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5), imKernel(3, 3);
    im.allocateImage();
    im2.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 -6 -6 -6 0 \
				 -6 6  3  6 -6\
				 -6 3  0  3 -6\
				 -6 6  3  6 -6\
				 0 -6 -6 -6 0";
    std::istringstream st2(s2);
    st2 >> im2;

    std::string sk = "0 -1 0 \
				 -1 4 -1 \
				 0 -1 0";
    std::istringstream stk(sk);
    stk >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeMirrored);

    t_ImBorderSetConstant(im, (INT32) 200);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImConvolve<Image<INT32>, INT32, Image<INT32>>(im, imSE, 0, 1,
                                                             imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }
  static void testGradient2()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testGradient2");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5), imKernel(3, 3);
    im.allocateImage();
    im2.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "-3 -6 -9 -6 -3 \
				 -6 15  9 15 -6\
				 -9  9  0  9 -9\
				 -6 15  9 15 -6\
				 -3 -6 -9 -6 -3";
    std::istringstream st2(s2);
    st2 >> im2;

    std::string sk = "-1 -1 -1 \
				 -1  8 -1 \
				 -1 -1 -1";
    std::istringstream stk(sk);
    stk >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeClip);

    t_ImBorderSetConstant(im, (INT32) 200);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImConvolve(im, imSE, imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }
  // avec border_mirror
  static void testGradient2b()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("BasicConvolutionTests::testGradient2b");
    morphee::Image<INT32> im(5, 5), im2(5, 5), imOut(5, 5), imKernel(3, 3);
    im.allocateImage();
    im2.allocateImage();
    imKernel.allocateImage();
    imOut.allocateImage();

    std::string s =

        "0 0 0 0 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 3 3 3 0\
				  0 0 0 0 0";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "-12 -12 -18 -12 -12 \
				 -12 15  9 15 -12\
				 -18  9  0  9 -18\
				 -12 15  9 15 -12\
				 -12 -12 -18 -12 -12";
    std::istringstream st2(s2);
    st2 >> im2;

    std::string sk = "-1 -1 -1 \
				 -1  8 -1 \
				 -1 -1 -1";
    std::istringstream stk(sk);
    stk >> imKernel;

    selement::ConvolutionKernel<INT32> imSE(imKernel, Point3D(1, 1, 0),
                                            selement::SEBorderTypeMirrored);

    t_ImBorderSetConstant(im, (INT32) 200);

    BOOST_MESSAGE("\tImage In  : " << im);
    BOOST_MESSAGE("\tKernel    : " << imKernel);
    BOOST_MESSAGE("\tExpected  : " << im2);
    filters::t_ImConvolve<Image<INT32>, INT32, Image<INT32>>(im, imSE, 0, 1,
                                                             imOut);
    BOOST_MESSAGE("\tResult    : " << imOut);

    Image<INT32>::iterator itIn, itEnd, itOut;
    for (itIn = im2.begin(), itEnd = im2.end(), itOut = imOut.begin();
         itIn != itEnd; ++itIn, ++itOut)
      BOOST_CHECK(*itIn == *itOut);
  }
  static void testGradient3()
  {
    s_logCleaner cleaner;
    /*
    BOOST_MESSAGE("BasicConvolutionTests::testGradient3");
    ImageInterface *_orig;
    RES_C res=imageIO::rasterFileRead(_orig,"/home/romain/images/foreman.ras");
    if(res != RES_OK)
      MORPHEE_UNWIND_STACK();
    BOOST_REQUIRE(res==RES_OK);
    Image<UINT8>*orig=dynamic_cast<Image<UINT8>*>(_orig);
    Image<INT16> tmp=orig->t_getSame<INT16>();
    Image<INT16> out=tmp.getSame();

    t_ImCopy(*orig,tmp);

    Image<INT16>imKernel(3,3);
    imKernel.allocateImage();
    std::string sk=
      "-1 -1 -1 \
       -1 8 -1 \
       -1 -1 -1";
    std::istringstream stk(sk);
    stk>>imKernel;

    selement::ConvolutionKernel<INT16>gradientKernel(imKernel,1,1,0);

    filters::t_ImConvolve<INT16,INT16,INT16>(tmp,gradientKernel, 128,1,out);



    Image<UINT8> out8=out.t_getSame<UINT8>();
    t_ImBoundedCopy<INT16,UINT8>(out,0,255,out8);
    imageIO::pngFileWrite(&out8,"/tmp/foremanGrad8.png")==RES_OK;

    std::string sk4=
      "0 -1 0 \
       -1 4 -1 \
       0 -1 0";
    std::istringstream stk4(sk4);
    stk4>>imKernel;
    selement::ConvolutionKernel<INT16>gradientKernel4(imKernel,1,1,0);

    filters::t_ImConvolve(tmp,gradientKernel4,(INT16)128,(INT16)1,out);
    t_ImBoundedCopy<INT16,UINT8>(out,0,255,out8);
    imageIO::pngFileWrite(&out8,"/tmp/foremanGrad4.png")==RES_OK;
    */
  }
}; // class BasicConvolutionTests

class GaussianTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&testCreateGaussianKernel3));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testCreateGaussianKernel7));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGaussian1));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGaussian3D));
    filtersGlobalTest->add(BOOST_TEST_CASE(&regression1));
    filtersGlobalTest->add(BOOST_TEST_CASE(&regressionLargeGaussian));
  }

  static void testCreateGaussianKernel3()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("GaussianTests::testCreateGaussianKernel:");
    Image<F_DOUBLE> kIm(3, 3);
    kIm.allocateImage();

    // pour faire taire valgrind:
    t_ImSetConstant(kIm, 0);
    selement::ConvolutionKernel<F_DOUBLE> k(kIm, Point3D(1, 1, 0));

    filters::t_fillGaussianKernel(k);

    BOOST_MESSAGE("\tgaussian kernel" << k);

    selement::ConvolutionKernel<F_DOUBLE>::const_iterator it, iend;

    F_DOUBLE sum = 0.;
    for (it = k.begin(), iend = k.end(); it != iend; ++it) {
      sum += *it;
    }
    BOOST_MESSAGE("\tsomme: " << sum);
    F_DOUBLE eps_factor = 1.e0;
    BOOST_CHECK_MESSAGE(
        std::abs(sum - 1.) <=
            eps_factor * std::numeric_limits<F_DOUBLE>::epsilon(),
        sum << " pas proche de 1 (à "
            << eps_factor * std::numeric_limits<F_DOUBLE>::epsilon()
            << " près )");
  }
  static void testCreateGaussianKernel7()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("GaussianTests::testCreateGaussianKernel:");
    Image<F_DOUBLE> kIm(7, 7);
    kIm.allocateImage();

    // pour faire taire valgrind:
    t_ImSetConstant(kIm, 0);
    selement::ConvolutionKernel<F_DOUBLE> k(kIm, Point3D(3, 3, 0));

    filters::t_fillGaussianKernel(k);

    BOOST_MESSAGE("\tgaussian kernel" << k);

    selement::ConvolutionKernel<F_DOUBLE>::const_iterator it, iend;

    F_DOUBLE sum = 0;
    for (it = k.begin(), iend = k.end(); it != iend; ++it) {
      sum += *it;
    }
    BOOST_MESSAGE("\tsomme: " << sum);
    F_DOUBLE eps_factor = 5.e0;
    BOOST_CHECK_MESSAGE(std::abs(sum - 1.) <=
                            eps_factor *
                                std::numeric_limits<F_DOUBLE>::epsilon(),
                        sum << " pas proche de 1");
  }

  static void testGaussian1()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("GaussianTests::testGaussian1:");
    Image<F_DOUBLE> im(5, 5);
    im.allocateImage();

    std::string s = "0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 9. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.";

    std::stringstream sst(s);
    sst >> im;

    Image<F_DOUBLE> imOut = im.getSame();
    imOut.allocateImage();

    Image<F_DOUBLE> imTheory = im.getSame();
    imTheory.allocateImage();

    BOOST_MESSAGE("imIn:     " << im);

    const F_DOUBLE *raw_theory = imTheory.rawPointer();
    const F_DOUBLE *raw_out    = imOut.rawPointer();
    BOOST_CHECK(filters::t_ImGaussianFilter(im, 1, imOut) == RES_OK);
    BOOST_CHECK(filters::t_ImGaussianFilter_Slow(im, 1, imTheory) == RES_OK);

    BOOST_REQUIRE(imTheory.rawPointer() == raw_theory);
    BOOST_REQUIRE(imOut.rawPointer() == raw_out);

    BOOST_MESSAGE("imTheory: " << imTheory);
    BOOST_MESSAGE("imOut:    " << imOut);

    Image<F_DOUBLE>::const_iterator itOut    = imOut.begin(),
                                    iendOut  = imOut.end(),
                                    itTheory = imTheory.begin();

    for (; itOut != iendOut; ++itOut, ++itTheory) {
      BOOST_CHECK_MESSAGE(std::abs(*itOut - *itTheory) < 1.e-5,
                          *itOut << " != " << *itTheory << " (expected) : pos "
                                 << itTheory.Position());
    }
  }
  static void testGaussian3D()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("GaussianTests::testGaussian1:");
    Image<F_DOUBLE> im(5, 5, 5);
    im.allocateImage();

    std::string s = "0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
								   \
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
								   \
			               0. 0. 0. 0. 0.\
			               0. 0. 0. 0. 0.\
						   0. 0. 9. 0. 0.\
			               0. 0. 0. 0. 0.\
			               0. 0. 0. 0. 0.\
						           \
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
								   \
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.\
						   0. 0. 0. 0. 0.";

    BOOST_REQUIRE(t_ImDimension(im) == 3);

    std::stringstream sst(s);
    sst >> im;

    Image<F_DOUBLE> imOut = im.getSame();
    imOut.allocateImage();

    Image<F_DOUBLE> imTheory = im.getSame();
    imTheory.allocateImage();

    BOOST_MESSAGE("imIn:     " << im);
    BOOST_CHECK(filters::t_ImGaussianFilter(im, 1, imOut) == RES_OK);
    BOOST_CHECK(filters::t_ImGaussianFilter_Slow(im, 1, imTheory) == RES_OK);
    BOOST_MESSAGE("imTheory: " << imTheory);
    BOOST_MESSAGE("imOut:    " << imOut);

    Image<F_DOUBLE>::const_iterator itOut    = imOut.begin(),
                                    iendOut  = imOut.end(),
                                    itTheory = imTheory.begin();

    for (; itOut != iendOut; ++itOut, ++itTheory) {
      BOOST_CHECK_MESSAGE(std::abs(*itOut - *itTheory) < 1.e-5,
                          *itOut << " != " << *itTheory << " (expected)");
    }
  }

  static void regression1()
  {
    s_logCleaner cleaner;
    // I have found a dark border around the filter output,
    // so I check that a flat 127 image won't have that.

    BOOST_MESSAGE("GaussianTests::regression1");
    int imSize = 5;
    Image<F_DOUBLE> im(imSize, imSize);
    im.allocateImage();

    Image<F_DOUBLE> imOut(imSize, imSize);
    imOut.allocateImage();

    F_DOUBLE dVal = 127.;
    t_ImSetConstant(im, dVal);

    filters::t_ImGaussianFilter(im, 3, imOut);

    Image<F_DOUBLE>::iterator it, iend;

    BOOST_MESSAGE("\tinput: " << im);
    BOOST_MESSAGE("\toutput: " << imOut);

    for (it = imOut.begin(), iend = imOut.end(); it != iend; ++it) {
      BOOST_CHECK_MESSAGE(fabs(*it - dVal) <= 0.001, *it << "!=" << dVal << "("
                                                         << it.getX() << ","
                                                         << it.getY() << ")");
    }
  }

  static void regressionLargeGaussian()
  {
    s_logCleaner cleaner;
    coord_t sz = 12;
    Image<F_DOUBLE> im(sz, sz);
    im.allocateImage();

    BOOST_REQUIRE(t_ImSetConstant(im, (F_DOUBLE) 0.) == RES_OK);

    Image<F_DOUBLE> imG = im.getSame();

    BOOST_CHECKPOINT("large gaussian, point 1");

    filters::t_ImGaussianFilter(im, sz, imG);
  }

}; // GaussianTests

class RecursiveGaussianTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&test1));
  }

  static void test1()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("FastGaussianTests::test1");

    Image<F_DOUBLE> im(7, 7);
    im.allocateImage();
    std::string sk = "0 0 0 0 0 0 0\
				 0 0 0 0 0 0 0\
				 0 0 0 0 0 0 0\
				 0 0 0 255 0 0 0\
				 0 0 0 0 0 0 0\
				 0 0 0 0 0 0 0\
				 0 0 0 0 0 0 0";
    std::istringstream stk(sk);
    stk >> im;

    Image<INT32> imInt(7, 7);
    imInt.allocateImage();

    Image<F_DOUBLE> im2 = im.getSame();

    BOOST_REQUIRE(filters::ImGaussianRecursive_Helper(im, 3, im2) == RES_OK);

    BOOST_MESSAGE("input image:  \t" << im);
    t_ImCopy(im2, imInt);
    BOOST_MESSAGE("output image: \t" << imInt);
    BOOST_MESSAGE("output image: \t" << im2);

    F_DOUBLE sum        = 0;
    F_DOUBLE eps_factor = 5.e0;
    Image<F_DOUBLE>::iterator it, iend;

    for (it = im2.begin(), iend = im2.end(); it != iend; ++it) {
      sum += *it;
    }
    BOOST_CHECK_MESSAGE(std::abs(sum - 255.) <=
                            eps_factor *
                                std::numeric_limits<F_DOUBLE>::epsilon(),
                        sum << " pas proche de 255");

    BOOST_REQUIRE(filters::t_ImGaussianFilter(im, 1, im2) == RES_OK);
    sum = 0.;
    for (it = im2.begin(), iend = im2.end(); it != iend; ++it) {
      sum += *it;
    }
    BOOST_CHECK_MESSAGE(std::abs(sum - 255.) <=
                            eps_factor *
                                std::numeric_limits<F_DOUBLE>::epsilon(),
                        sum << " pas proche de 255");

    t_ImCopy(im2, imInt);
    BOOST_MESSAGE("output image: \t" << imInt);
    BOOST_MESSAGE("output image: \t" << im2);
  }

}; // RecursiveGaussianTests

extern test_suite *filtersGlobalTest;
void test_Convolve()
{
  BasicConvolutionTests::addTests();
  GaussianTests::addTests();
  // RecursiveGaussianTests::addTests();
}
