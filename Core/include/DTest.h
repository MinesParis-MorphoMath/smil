/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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

#ifndef _DTEST_H
#define _DTEST_H

#include <ctime>
#include <iostream>
#include <list>
#include <sstream>

#include "private/DTypes.hpp"
#include "DCommon.h"
#include "DTime.h"
#include "DErrors.h"

using namespace std;

namespace smil
{
  class TestCase
  {
  public:
    TestCase() : stopIfError(false), outStream(NULL)
    {
    }
    virtual ~TestCase()
    {
    }
    virtual void init()
    {
    }
    virtual void run() = 0;
    virtual void end()
    {
    }
    const char *name;
    bool stopIfError;
    stringstream *outStream;
    RES_T retVal;
#ifdef __clang__
    SMIL_UNUSED
#endif // __clang__
    int tElapsed;
  };

#define TEST_ASSERT(expr)                                                      \
  {                                                                            \
    if (!(expr)) {                                                             \
      if (outStream)                                                           \
        *outStream << __FILE__ << ":" << __LINE__ << ": error: "               \
                   << " assert " << #expr << endl;                             \
      retVal = RES_ERR;                                                        \
      if (stopIfError)                                                         \
        return;                                                                \
    }                                                                          \
  }

#define TEST_NO_THROW(expr)                                                    \
  {                                                                            \
    bool _throw = false;                                                       \
    try {                                                                      \
      expr;                                                                    \
    } catch (...) {                                                            \
      _throw = true;                                                           \
    }                                                                          \
    if (_throw) {                                                              \
      if (outStream)                                                           \
        *outStream << __FILE__ << ":" << __LINE__ << ": error: "               \
                   << " no throw " << #expr << endl;                           \
      retVal = RES_ERR;                                                        \
      if (stopIfError)                                                         \
        return;                                                                \
    }                                                                          \
  }

#define TEST_THROW(expr)                                                       \
  {                                                                            \
    bool _throw = false;                                                       \
    try {                                                                      \
      expr;                                                                    \
    } catch (...) {                                                            \
      _throw = true;                                                           \
    }                                                                          \
    if (!_throw) {                                                             \
      if (outStream)                                                           \
        *outStream << __FILE__ << ":" << __LINE__ << ": error: "               \
                   << " throw " << #expr << endl;                              \
      retVal = RES_ERR;                                                        \
      if (stopIfError)                                                         \
        return;                                                                \
    }                                                                          \
  }

  class TestSuite
  {
  public:
    void add(TestCase *f);
    int run();

  private:
    list<TestCase *> funcList;
#ifdef __clang__
    SMIL_UNUSED
#endif // __clang__
    int tElapsed;
  };

#define ADD_TEST(TS, TC)                                                       \
  TC TC##_inst;                                                                \
  TC##_inst.name = #TC;                                                        \
  TS.add(&TC##_inst);

} // namespace smil

#endif // _DTEST_H
