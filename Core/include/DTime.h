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


#ifndef _DTIME_H
#define _DTIME_H

#include <time.h>


#ifdef _MSC_VER
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif 

namespace smil
{
    #ifdef _MSC_VER

    typedef struct timeval {
	UINT64 tv_sec;
	UINT64 tv_usec;
    } timeval;

    inline int gettimeofday (struct timeval *tp, void *tz)
    {
	struct _timeb timebuffer;
	_ftime (&timebuffer);
	tp->tv_sec = timebuffer.time;
	tp->tv_usec = timebuffer.millitm * 1000;
	return 0;
    }

    #endif // _MSC_VER

	
    static inline double getCpuTime()
    {
	struct timeval tv;
	if (gettimeofday(&tv, 0)) 
	{
	    cout << "gettimeofday returned error" << endl;
	}
	return tv.tv_sec + tv.tv_usec/1000000;
    }

    inline string displayTime(double tSec)
    {
	stringstream s;
	
	if (tSec>=1.)
	  s << int(tSec*1E3)/1E3 << " secs";
	else if (tSec*1E3>=1.)
	  s << int(tSec*1E6)/1E3 << " msecs";
	else 
	  s << int(tSec*1E6) << " usecs";
	
	return s.str();
    }


} // namespace smil

#endif // _DTIME_H

