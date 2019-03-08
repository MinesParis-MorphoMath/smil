
#include <boost/test/unit_test.hpp>
using namespace boost::unit_test_framework;
test_suite *filtersMainTests(int argc, char *argv[]);

#ifdef BOOST_TEST_DYN_LINK
// this section might also depend on MORPHEE_BOOST_VERSION_MINOR >= 34 and
// also on the Darwin (MacOSX) platform as it has been written to be used on
// MacOSX Tiger(Intel) with Boost 1.34

bool init_function()
{
  // dummy
  return true;
}

int main(int argc, char *argv[])
{
  framework::master_test_suite().add(filtersMainTests(argc, argv));
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else

test_suite *init_unit_test_suite(int argc, char *argv[])
{
  return filtersMainTests(argc, argv);
}

#endif
