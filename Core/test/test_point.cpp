/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "DImage.h"
#include "DTest.h"

#include <iostream>
#include <fstream>

using namespace smil;

class Test_Add : public TestCase
{
  virtual void run()
  {
    Point<off_t> pr(25, 45, 65);
    Point<off_t> p1(10, 20, 30);
    Point<off_t> p2(15, 25, 35);

    Point<off_t> p3 = p1 + p2;

    TEST_ASSERT(p3 == pr);
  }
};

class Test_Sub : public TestCase
{
  virtual void run()
  {
    Point<off_t> pr(5, 0, -5);
    Point<off_t> p1(15, 20, 30);
    Point<off_t> p2(10, 20, 35);

    Point<off_t> p3 = p1 - p2;

    TEST_ASSERT(p3 == pr);
  }
};


int main()
{
  TestSuite ts;

  ADD_TEST(ts, Test_Add);
  ADD_TEST(ts, Test_Sub);

  return ts.run();
}
