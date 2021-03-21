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
    double rExpect = 0.641;

    double r = indexAccuracy(imGt, imIn);

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
    double rExpect = 0.563;

    double r = indexSensitivity(imGt, imIn);

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
    double rExpect = 0.667;

    double r = indexSpecificity(imGt, imIn);

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
    double rExpect = 0.333;

    double r = indexFallOut(imGt, imIn);

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
    double rExpect = 0.437;

    double r = indexMissRate(imGt, imIn);

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
    double rExpect = 0.360;

    double r = indexPrecision(imGt, imIn);

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
    double rExpect = 0.563;

    double r = indexRecall(imGt, imIn);

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
    double rExpect = 0.439;

    double r = indexFScore(imGt, imIn);

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
    double rExpect = 0.563;

    double r = indexOverlap(imGt, imIn);

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
    double rExpect = 0.281;

    double r = indexJaccard(imGt, imIn);

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
    double rExpect = 23552;

    double r = distanceHamming(imGt, imIn);

    bool ok = abs(r - rExpect) < 1;
    TEST_ASSERT(ok);
    if (!ok)
      cout << "\n\t distanceHamming \t" << rExpect << "\t" << r << endl;
  }
};

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

int main(void)
{
  TestSuite ts;

  off_t m, M;

  m = dim/8;
  M = 5*dim/8;

  for (off_t j = m; j < M; j++)
    for (off_t i = m; i < M; i++)
      imGt.setPixel(i, j, 255);

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
  ADD_TEST(ts, Test_Hausdorff);

  return ts.run();
}
