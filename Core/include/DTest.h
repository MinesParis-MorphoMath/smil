/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
 *     * Neither the name of the University of California, Berkeley nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _DTEST_H
#define _DTEST_H

#include <time.h>
#include <iostream>
#include <list>
#include <sstream>

#include "DTypes.hpp"

using namespace std;

inline const char *displayTime(int t)
{
    stringstream s;
    double tSec = double(t) / CLOCKS_PER_SEC;
    
    if (int(tSec)!=0 || tSec==0)
      s << tSec << " secs";
    else if (int(tSec*1E3)!=0)
      s << tSec*1E3 << " msecs";
    else 
      s << tSec*1E6 << " µsecs";
    
    return s.str().c_str();
}

class TestCase
{
public:
  TestCase() 
    : stopIfError(false), 
      outStream(NULL) 
  {
  }
  virtual void init() {}
  virtual void run() = 0;
  virtual void end() {}
  const char *name;
  stringstream *outStream;
  RES_T retVal;
  int tElapsed;
  bool stopIfError;
};


#define TEST_ASSERT(expr) \
{ \
    if (!(expr)) \
    { \
	    if (outStream) \
		*outStream << __FILE__ << ":" <<  __LINE__ << ": error: " << " assert " << #expr << endl;	\
	    retVal = RES_ERR; \
	    if (stopIfError) \
	      return; \
    } \
}

#define TEST_NO_THROW(expr) \
{ \
    bool _throw = false; \
    try { expr; } \
    catch(...) { _throw = true; } \
    if (_throw) \
    { \
	    if (outStream) \
		*outStream << __FILE__ << ":" <<  __LINE__ << ": error: " << " no throw " << #expr << endl;	\
	    retVal = RES_ERR; \
	    if (stopIfError) \
	      return; \
    } \
}

#define TEST_THROW(expr) \
{ \
    bool _throw = false; \
    try { expr; } \
    catch(...) { _throw = true; } \
    if (!_throw) \
    { \
	    if (outStream) \
		*outStream << __FILE__ << ":" <<  __LINE__ << ": error: " << " throw " << #expr << endl;	\
	    retVal = RES_ERR; \
	    if (stopIfError) \
	      return; \
    } \
}

class TestSuite
{
public:
  void add(TestCase *f);
  RES_T run();
private:
  list<TestCase*> funcList;
  int tElapsed;
};


#define ADD_TEST(TS, TC) \
TC TC##_inst; \
TC##_inst.name = #TC; \
TS.add(& TC##_inst); 



#endif // _DTEST_H

