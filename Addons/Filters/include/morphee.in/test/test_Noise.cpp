#include <morphee/common/include/commonError.hpp>
#include <morphee/imageIO/include/morpheeImageIO.hpp>
#include <morphee/image/include/imageArithmetic.hpp>
#include <morphee/image/include/private/image_T.hpp>

#include <morphee/filters/include/private/filtersNoise_T.hpp>

// This test program requires Boost::Test
#include <boost/test/unit_test.hpp>
using namespace boost::unit_test_framework;

extern test_suite *filtersGlobalTest;

using namespace morphee;
class NoiseTests
{
public:
  static void addTests()
  {
    filtersGlobalTest->add(BOOST_TEST_CASE(&testSaltAndPepper));
    // filtersGlobalTest->add(BOOST_TEST_CASE( &testNormalLaw ));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGaussian));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testGaussianRGB));
    filtersGlobalTest->add(BOOST_TEST_CASE(&testPoissonian));
  }

  static void testSaltAndPepper()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("NoiseTests::testSaltAndPepper:");
    Image<UINT8> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s = "1 1 1 1 1\
				  1 1 1 1 1\
				  1 1 1 1 1\
				  1 1 1 1 1\
				  1 1 1 1 1";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "0 1 1 1 255\
				  1 255 255 1 1\
				  1 1 1 1 1\
				  1 0 1 1 1\
				  1 255 1 255 1";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("input:   \t" << im);
    BOOST_MESSAGE("expected:\t" << im2);
    // Always have the same distribution
    filters::setSeed(1u);
    filters::t_ImAddNoiseSaltAndPepper(im, 0.3, imOut);
    BOOST_MESSAGE("real:    \t" << imOut);
    BOOST_MESSAGE(
        "  (Note that actual results depend on the drand48 implementation)");

    Image<UINT8>::const_iterator it2, iend2, itO;

    for (it2 = im2.begin(), iend2 = im2.end(), itO = imOut.begin();
         it2 != iend2; ++it2, ++itO) {
      BOOST_CHECK_MESSAGE(*it2 == *itO,
                          (int) *itO << " != " << (int) *it2 << "(expected)");
    }
  }

#if 0
		// Highly unportable test, just to check that my normal law is indeed normal ;)
		static void testNormalLaw()
		{
			const unsigned int numClasses=21;
			std::vector<unsigned int>freqTab(numClasses,0);

			unsigned int numIter=1000;
			filters::setSeed(0);

			unsigned int freqTabTheory[numClasses]={2,1,10,12,25,31,49,80,109,125,117,121,116,80,48,26,24,14,5,2,3 };

			for(unsigned int i=0; i<numIter; ++i)
			{
				int index=(int)filters::normalDistribution(5.,10.5); // mean: 10, std dev: 5.
				if(index < 0)
					index=0;
				if(index >= (int)freqTab.size())
					index = freqTab.size()-1;

				freqTab[index]++;
			}

			for(unsigned int i=0; i<numClasses;++i)
			{
				BOOST_MESSAGE("value "<<i<<"  freq: "<<freqTab[i]<<"  freqTheory:"<<freqTabTheory[i]);
				BOOST_CHECK_MESSAGE(freqTab[i]==freqTabTheory[i],freqTab[i]<<"!="<<freqTabTheory[i]<<"(expected");
			}

		}
#endif
  static void testGaussian()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("NoiseTests::testGaussian:");
    Image<UINT8> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s = "127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "131 127 110 114 133\
				  125 116 126 125 143\
				  137 133 111 118 116\
				  120 150 104 123 101\
				  123 137 138 118 147";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("input:   \t" << im);
    BOOST_MESSAGE("expected:\t" << im2);
    // Always have the same distribution
    filters::setSeed(1);
    filters::t_ImAddNoiseGaussian(im, 10., imOut);
    BOOST_MESSAGE("real:    \t" << imOut);
    BOOST_MESSAGE(
        "  (Note that actual results depend on the drand48 implementation)");

    Image<UINT8>::const_iterator it2, iend2, itO;

    for (it2 = im2.begin(), iend2 = im2.end(), itO = imOut.begin();
         it2 != iend2; ++it2, ++itO) {
      BOOST_CHECK_MESSAGE(*it2 == *itO,
                          (int) *itO << " != " << (int) *it2 << "(expected)");
    }
  }

  static void testGaussianRGB()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("NoiseTests::testGaussian:");
    Image<pixel_3<UINT8>> im(2, 2), im2(2, 2), imOut(2, 2);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s = "127 127 127   127 127 127\
				  127 127 127   127 127 127";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "131 127 110   114 133 125\
				  116 126 125   143 137 133";
    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("input:   \t" << im);
    BOOST_MESSAGE("expected:\t" << im2);
    // Always have the same distribution
    filters::setSeed(1);
    filters::t_ImAddNoiseGaussian(im, 10., imOut);
    BOOST_MESSAGE("real:    \t" << imOut);
    BOOST_MESSAGE(
        "  (Note that actual results depend on the drand48 implementation)");

    Image<pixel_3<UINT8>>::const_iterator it2, iend2, itO;

    for (it2 = im2.begin(), iend2 = im2.end(), itO = imOut.begin();
         it2 != iend2; ++it2, ++itO) {
      BOOST_CHECK_MESSAGE(*it2 == *itO, *itO << " != " << *it2 << "(expected)");
    }
  }

  static void testPoissonian()
  {
    s_logCleaner cleaner;
    BOOST_MESSAGE("NoiseTests::testPoissonnian:");
    Image<UINT8> im(5, 5), im2(5, 5), imOut(5, 5);
    im.allocateImage();
    im2.allocateImage();
    imOut.allocateImage();

    std::string s = "127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127\
				  127 127 127 127 127";
    std::istringstream st(s);
    st >> im;

    std::string s2 = "119 133 123 118 135\
 				 128 137 114 118 117\
				 127 132 108 121 135\
 				 123 134 128 142 134\
 				 130 110 113 121 134";

    std::istringstream st2(s2);
    st2 >> im2;

    BOOST_MESSAGE("input:   \t" << im);
    BOOST_MESSAGE("expected:\t" << im2);
    // Always have the same distribution
    filters::setSeed(1);
    filters::t_ImAddNoisePoissonian(im, imOut);
    BOOST_MESSAGE("real:    \t" << imOut);
    BOOST_MESSAGE(
        "  (Note that actual results depend on the drand48 implementation)");

    Image<UINT8>::const_iterator it2, iend2, itO;

    for (it2 = im2.begin(), iend2 = im2.end(), itO = imOut.begin();
         it2 != iend2; ++it2, ++itO) {
      BOOST_CHECK_MESSAGE(*it2 == *itO,
                          (int) *itO << " != " << (int) *it2 << "(expected)");
    }
  }

}; // class NoiseTests

void test_Noise()
{
  NoiseTests::addTests();
}
