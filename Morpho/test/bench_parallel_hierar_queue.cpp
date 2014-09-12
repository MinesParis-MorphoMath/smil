/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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



#include "Core/include/DCore.h"
#include "DMorpho.h"

using namespace smil;

template <class T>
RES_T fill_hq (const Image<T> &img, HierarchicalQueue<T> &hq) {
	hq.initialize (img);
	typename ImDtTypes<T>::lineType inPixels = img.getPixels ();
	size_t s[3];
	img.getSize (s);
	size_t offset = 0;
	for (size_t i=0; i<img.getPixelCount(); ++i){
		hq.push (*inPixels, offset);
		inPixels++;
		offset++;
	}
	return RES_OK;
}

template <class T>
RES_T fill_par_hq (const Image<T> &img, ParHierarQInc<T> hq) {
	hq.initialize_and_fill (img);
	return RES_OK;
}

int main(int argc, char *argv[])
{
    Image_UINT8 img("http://cmm.ensmp.fr/~faessel/smil/images/barbara.png");
    HierarchicalQueue<UINT8> hq;
    ParHierarQInc<UINT8> hq2;    

    UINT BENCH_NRUNS = 10;
    
    BENCH_IMG(fill_hq, img, hq);
    BENCH_IMG(fill_par_hq, img, hq2);
    
        
}

