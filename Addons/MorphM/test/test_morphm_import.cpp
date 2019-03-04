
#include "DMorphMImage.hpp"
#include "Core/include/DCore.h"

#include <morphee/image/include/imageManipulation.hpp>
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageManipulation_T.hpp>
#include <morphee/image/include/private/imageArithmetic_T.hpp>

using namespace smil;

namespace morphee
{
  // Unnecessary but unresolved morphm functions (?)
  std::string ImageDispatchInfoLogger::get_latest_failing_dispatch_table()
  {
  }
  void ImageDispatchInfoLogger::set_latest_failing_dispatch_table(
      const std::string &dispatch_table_str)
  {
  }
  std::ostream &operator<<(std::ostream &, morphee::colorInfo const &)
  {
  }

} // namespace morphee

class Test_MorphM_Import : public TestCase
{
  virtual void run()
  {
    morphee::Image<UINT8> mIm(4, 5);
    mIm.allocateImage();

    std::string sA = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20";
    std::istringstream imstreamTheory(sA);
    imstreamTheory >> mIm;

    MorphmInt<UINT8> mInt(mIm);

    UINT8 vec1[20] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    Image<UINT8> sIm(4, 5);
    sIm << vec1;

    TEST_ASSERT(mInt == sIm);

    if (retVal != RES_OK) {
      mInt.printSelf(1);
      sIm.printSelf(1);
    }
  }
};

int main(int argc, char *argv[])
{
  TestSuite ts;
  ADD_TEST(ts, Test_MorphM_Import);

  return ts.run();
}
