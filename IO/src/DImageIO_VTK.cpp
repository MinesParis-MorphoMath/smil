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


#include "IO/include/private/DImageIO.hpp"
#include "IO/include/private/DImageIO_VTK.hpp"


namespace smil
{
    inline int splitStr(const std::string &s, char delim, std::vector<std::string> &elems) 
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
    
    RES_T readVTKHeader(ifstream &fp, VTKHeader &hStruct)
    {
        std::string buf, wrd;
        std::vector<std::string> bufElems;

        hStruct.scalarType = ImageFileInfo::SCALAR_TYPE_UINT8; // default, if none specified
        
        while(getline(fp, buf))
        {
            // To uppercase
            transform(buf.begin(), buf.end(), buf.begin(), ::toupper);
            // Split
            splitStr(buf, ' ', bufElems);
            
            // Get the first word
            wrd = bufElems[0];
            // And the first char
            char fc = wrd[0];
            
            // Check if we reached the end of the header
            if (isdigit(fc) || fc=='-') // number
              break;
            else if (!isalnum(fc) && fc!='#') // binary data
              break;
            else 
              hStruct.startPos = fp.tellg();
            
            if (wrd=="DATASET")
            {
                if (bufElems[1]!="STRUCTURED_POINTS")
                {
                    cout << "Error: vtk file type " << bufElems[1] << " not supported (must be STRUCTURED_POINTS)" << endl;
                    return RES_ERR;
                }
            }
            else if (wrd=="ASCII")
              hStruct.binaryFile = false;
            else if (wrd=="BINARY")
              hStruct.binaryFile = true;
            
            else if (wrd=="DIMENSIONS")
            {
                hStruct.width = atoi(bufElems[1].c_str());
                hStruct.height = atoi(bufElems[2].c_str());
                hStruct.depth = atoi(bufElems[3].c_str());
            }
            else if (wrd=="POINT_DATA")
                hStruct.pointNbr = atoi(bufElems[1].c_str());
            else if (wrd=="SCALARS")
            {
                hStruct.scalarTypeStr = bufElems[2];
                if (hStruct.scalarTypeStr=="UNSIGNED_CHAR")
                  hStruct.scalarType = ImageFileInfo::SCALAR_TYPE_UINT8;
                else if (hStruct.scalarTypeStr=="UNSIGNED_SHORT")
                  hStruct.scalarType = ImageFileInfo::SCALAR_TYPE_UINT16;                
                else if (hStruct.scalarTypeStr=="SHORT")
                  hStruct.scalarType = ImageFileInfo::SCALAR_TYPE_INT16;                
                else
                  hStruct.scalarType = ImageFileInfo::SCALAR_TYPE_UNKNOWN;                
            }
            else if (wrd=="COLOR_SCALARS")
            {
                hStruct.scalarCoeff = atoi(bufElems[2].c_str());
            }
        }
        
        return RES_OK;
    }
    
    RES_T getVTKFileInfo(const char* /*filename*/, ImageFileInfo &/*fInfo*/)
    {
        return RES_OK;
    }
    
} // namespace smil

