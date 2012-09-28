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
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
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


#ifndef _DBENCH_H
#define _DBENCH_H

#include "sys/time.h"

#ifdef _MSC_VER

// Work-around to MSVC __VA_ARGS__ expanded as a single argument, instead of being broken down to multiple ones
#define EXPAND( x ) x
#define _FIRST_VA_ARG(arg0, ...) arg0
#define FIRST_VA_ARG(x) EXPAND(_FIRST_VA_ARG(x))

#define _xPP_NARGS_IMPL(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,N,...) N
#define PP_NARGS(...) \
    EXPAND(_xPP_NARGS_IMPL(__VA_ARGS__,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

#else // _MSC_VER

#define FIRST_VA_ARG(arg0, ...) arg0

#define _xPP_NARGS_IMPL(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,N,...) N
#define PP_NARGS(...) \
    _xPP_NARGS_IMPL(__VA_ARGS__,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

#endif // _MSC_VER


#define _CLOCK omp_get_wtime()
#define D_TIMEVAL(t1, t2) double(t2.tv_sec+t2.tv_usec/1e6-(t1.tv_sec+t1.tv_usec/1e6))
    
    
#define BENCH(func, ...) \
{ \
      struct timeval t1,t2; \
      gettimeofday(&t1,0); \
      for (UINT i=0;i<BENCH_NRUNS;i++) \
	  func(__VA_ARGS__); \
      gettimeofday(&t2,0); \
      cout << #func << "\t" << displayTime(D_TIMEVAL(t1, t2)/BENCH_NRUNS) << endl; \
}

#define BENCH_STR(func, str, ...) \
{ \
      struct timeval t1,t2; \
      gettimeofday(&t1,0); \
      for (UINT i=0;i<BENCH_NRUNS;i++) \
		func(__VA_ARGS__); \
      gettimeofday(&t2,0); \
      cout << #func << " " << str << "\t" << displayTime(D_TIMEVAL(t1, t2)/BENCH_NRUNS) << endl; \
}

#define BENCH_IMG(func, ...) \
{ \
      struct timeval t1,t2; \
      gettimeofday(&t1,0); \
      for (UINT i=0;i<BENCH_NRUNS;i++) \
		func(__VA_ARGS__); \
      gettimeofday(&t2,0); \
      cout << #func << "\t" << FIRST_VA_ARG(__VA_ARGS__).getTypeAsString() << "\t"; \
      cout << FIRST_VA_ARG(__VA_ARGS__).getWidth() << "x" << FIRST_VA_ARG(__VA_ARGS__).getHeight(); \
      if (FIRST_VA_ARG(__VA_ARGS__).getDepth()>UINT(1)) cout << "x" << FIRST_VA_ARG(__VA_ARGS__).getDepth(); \
      cout << "\t" << displayTime(D_TIMEVAL(t1, t2)/BENCH_NRUNS) << endl; \
}

#define BENCH_IMG_STR(func, str, ...) \
{ \
      struct timeval t1,t2; \
      gettimeofday(&t1,0); \
      for (UINT i=0;i<BENCH_NRUNS;i++) \
		func(__VA_ARGS__); \
      gettimeofday(&t2,0); \
      cout << #func << " " << str << "\t" << FIRST_VA_ARG(__VA_ARGS__).getTypeAsString() << "\t"; \
      cout << FIRST_VA_ARG(__VA_ARGS__).getWidth() << "x" << FIRST_VA_ARG(__VA_ARGS__).getHeight(); \
      if (FIRST_VA_ARG(__VA_ARGS__).getDepth()>1) cout << "x" << FIRST_VA_ARG(__VA_ARGS__).getDepth(); \
      cout << "\t" << displayTime(D_TIMEVAL(t1, t2)/BENCH_NRUNS) << endl; \
}



#endif // _DBENCH_H

