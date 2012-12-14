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


#ifndef _D_IMAGE_IO_VTK_HPP
#define _D_IMAGE_IO_VTK_HPP


#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "Core/include/private/DTypes.hpp"
#include "Core/include/private/DImage.hpp"

using namespace std;


namespace smil
{
    /** 
    * \addtogroup IO
    */
    /*@{*/
  
    inline int split(const std::string &s, char delim, std::vector<std::string> &elems) 
    {
	std::stringstream ss(s);
	std::string item;
	int nbr = 0;
	elems.clear();
	while(std::getline(ss, item, delim)) 
	{
	    elems.push_back(item);
	}
	return nbr;
    }

    /**
    * VTK file read
    */
    template <class T>
    RES_T readVTK(const char *filename, Image<T> &image)
    {
	std::ifstream fp;

	/* open image file */
	fp.open(filename, ios_base::binary);
    //     fp.open(filename);
	if (!fp)
	{
	    cerr << "error: couldn't open " << filename << "!" << endl;
	    return RES_ERR;
	}

	std::string buf, wrd;
	std::vector<std::string> bufElems;
	streampos startPos;

	bool isAscii = false;
	size_t ptsNbr = 0;
	size_t width = 0, height = 0, depth = 0;
	double scalarCoeff = 1.0;
	
	while(getline(fp, buf))
	{
	    // To uppercase
	    transform(buf.begin(), buf.end(), buf.begin(), ::toupper);
	    // Split
	    split(buf, ' ', bufElems);
	    
	    // Get the first word
	    wrd = bufElems[0];
	    // And the first char
	    char fc = wrd[0];
    //	bool id = isalnum(fc);
    //	bool im = fc=='-';
    //	bool ia = isascii(fc);
	    
	    // Check if we reached the end of the header
	    if (isdigit(fc) || fc=='-') // number
	      break;
	    else if (!isalnum(fc) && fc!='#') // binary data
	      break;
	    else 
	      startPos = fp.tellg();
	    
	    if (wrd=="DATASET")
	    {
		if (bufElems[1]!="STRUCTURED_POINTS")
		{
		    cout << "Error: vtk file type " << bufElems[1] << " not supported (must be STRUCTURED_POINTS)" << endl;
		    fp.close();
		    return RES_ERR;
		}
	    }
	    else if (wrd=="ASCII")
	      isAscii = true;
	    else if (wrd=="BINARY")
	      isAscii = false;
	    else if (wrd=="DIMENSIONS")
	    {
		width = atoi(bufElems[1].c_str());
		height = atoi(bufElems[2].c_str());
		depth = atoi(bufElems[3].c_str());
	    }
	    else if (wrd=="POINT_DATA")
		ptsNbr = atoi(bufElems[1].c_str());
	    else if (wrd=="SCALARS")
	    {
		
	    }
	    else if (wrd=="COLOR_SCALARS")
	    {
		scalarCoeff = double(ImDtTypes<T>::max()) / atoi(bufElems[2].c_str());
	    }
	}
	
	image.setSize(width, height, depth);
	typename Image<T>::lineType pixels = image.getPixels();
	
	// Return to the begining of the data
	fp.seekg(startPos);
	
	if (isAscii)
	{
	    double val;
	    while(fp && --ptsNbr>0)
	    {
		fp >> val;
		*pixels++ = val*scalarCoeff;
	    }
	    if(fp)
	    {
		fp >> val;
		*pixels = val*scalarCoeff;
	    }
	}

	else
	{
	    // In binary version, values are written as unsigned chars
    // 	while(fp && --ptsNbr>0)
    // 	    fp.read((char*)pixels++, sizeof(char));
    // 	if (fp)
    // 	    fp.read((char*)pixels, sizeof(char));
	    fp.read((char*)pixels, sizeof(T)*ptsNbr);
	}

	fp.close();

	image.modified();
	
	return RES_OK;
    }

    /**
    * VTK file write
    */
    template <class T>
    RES_T writeVTK(const Image<T> &image, const char *filename, bool binary=true)
    {
	std::ofstream fp;
	
	/* open image file */
	fp.open(filename, ios_base::binary);
	if (!fp)
	{
	    cerr << "error: couldn't open " << filename << "!" << endl;
	    return RES_ERR;
	}

	int w = image.getWidth();
	int h = image.getHeight();
	int d = image.getDepth();
	int pixNbr = w*h*d;
	
	fp << "# vtk DataFile Version 3.0" << endl;
	fp << "vtk output" << endl;
	fp << "BINARY" << endl;
	fp << "DATASET STRUCTURED_POINTS" << endl;
	fp << "DIMENSIONS " << w << " " << h << " " << d << endl;
	fp << "SPACING 1.0 1.0 1.0" << endl;
	fp << "ORIGIN 0 0 0" << endl;
	fp << "POINT_DATA " << pixNbr << endl;
	
	if (binary)
	  fp << "COLOR_SCALARS ImageScalars 1" << endl;
	else
	{
	    cerr << "not implemented (todo..)" << endl;
	}
	
	typename Image<T>::lineType pixels = image.getPixels();
	
	if (binary)
	{
	  // todo : make this generic
	    fp.write((char*)pixels, sizeof(T)*pixNbr);
	}

	fp.close();
	
	return RES_OK;
    }

/*@}*/

} // namespace smil


#endif // _D_IMAGE_IO_VTK_HPP
