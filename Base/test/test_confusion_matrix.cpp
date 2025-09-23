/*
 * Smil
 * Copyright (c) 2011-2015 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */

#include "Core/include/DCore.h"
#include "Base/include/DBase.h"

using namespace smil;

const size_t dim =256;

Image<UINT8> imGt(dim, dim);
Image<UINT8> imIn(dim,dim);

class Test_Accuracy : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.641;

    double r = cTable.Accuracy();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexAccuracy \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Sensitivity : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.563;

    double r = cTable.Sensitivity();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexSensitivity \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Specificity : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.667;

    double r = cTable.Specificity();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexSpecificity \t" << rExpect << "\t" << r << endl;
  }
};

class Test_FallOut : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.333;

    double r = cTable.FallOut();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexFallOut \t" << rExpect << "\t" << r << endl;
  }
};

class Test_MissRate : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.437;

    double r = cTable.MissRate();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexMissRate \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Precision : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.360;

    double r = cTable.Precision();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexPrecision \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Recall : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.563;

    double r = cTable.Recall();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexRecall \t" << rExpect << "\t" << r << endl;
  }
};

class Test_FScore : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.439;

    double r = cTable.FScore(1.);

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexFScore \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Overlap : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.563;

    double r = cTable.Overlap();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexOverlap \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Jaccard : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 0.281;

    double r = cTable.Jaccard();

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t indexJaccard \t" << rExpect << "\t" << r << endl;
  }
};

class Test_Hamming : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    double rExpect = 23552;

    double r = distanceHamming(imGt, imIn);

    bool ok = abs(r - rExpect) < 1;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t distanceHamming \t" << rExpect << "\t" << r << endl;
  }
};

#if 0
class Test_Hausdorff : public TestCase
{
  virtual void run()
  {
    double rExpect = 90.510;

    double r = distanceHausdorff(imGt, imIn);

    bool ok = abs(r - rExpect) < 0.001;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t distanceHausdorff \t" << rExpect << "\t" << r << endl;
  }
};
#endif


class Test_All : public TestCase
{
  virtual void run()
  {
    ConfusionMatrix<UINT8> cTable(imGt, imIn);
    bool ok = true;
    double r, rExpect;

    rExpect = 0.641;
    r = cTable.Accuracy();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.563;
    r = cTable.Sensitivity();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.667;
    r = cTable.Specificity();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.333;
    r = cTable.FallOut();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.437;
    r = cTable.MissRate();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.360;
    r = cTable.Precision();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.563;
    r = cTable.Recall();
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.439;
    r = cTable.FScore(1.);
    ok = ok && abs(r - rExpect) < 0.001;

    rExpect = 0.563;
    r = cTable.Overlap();
    ok = ok && abs(r - rExpect) < 0.001;

    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t All indexes \t" << rExpect << "\t" << r << endl;
  }
};
int main(void)
{
  TestSuite ts;

  off_t m, M;

  m = dim/8;
  M = 5*dim/8;

  fill(imGt, UINT8(0));
  for (off_t j = m; j < M; j++)
    for (off_t i = m; i < M; i++)
      imGt.setPixel(i, j, 255);

  fill(imIn, UINT8(0));
  m = 2*dim/8;
  M = 7*dim/8;
  for (off_t j = m; j < M; j++)
    for (off_t i = m; i < M; i++)
      imIn.setPixel(i, j, 255);

  ADD_TEST(ts, Test_Accuracy);
  ADD_TEST(ts, Test_Sensitivity);
  ADD_TEST(ts, Test_Specificity);
  ADD_TEST(ts, Test_FallOut);
  ADD_TEST(ts, Test_MissRate);
  ADD_TEST(ts, Test_Precision);
  ADD_TEST(ts, Test_Recall);
  ADD_TEST(ts, Test_FScore);
  ADD_TEST(ts, Test_Overlap);
  ADD_TEST(ts, Test_Jaccard);
  ADD_TEST(ts, Test_Hamming);
//  ADD_TEST(ts, Test_Hausdorff);
  ADD_TEST(ts, Test_All);

  return ts.run();
}
