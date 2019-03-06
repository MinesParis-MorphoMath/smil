// This test program requires Boost::Test

#include <boost/test/unit_test.hpp>
using namespace boost::unit_test_framework;

#include <morphee/common/include/commonError.hpp>

test_suite *filtersGlobalTest = BOOST_TEST_SUITE("Filters Tests");

void test_Convolve();
void test_Noise();
void test_Differential();
void test_Diffusion();
void test_harris();

test_suite *filtersMainTests(int argc, char *argv[])
{
  // default logger to standard output
  morphee::GlobalTracer::instance()->setLogger(
      morphee::getDefaultStdErrLogger());

  test_Convolve();
  test_Noise();
  test_Differential();
  test_Diffusion();
  test_harris();

  return filtersGlobalTest;
}
