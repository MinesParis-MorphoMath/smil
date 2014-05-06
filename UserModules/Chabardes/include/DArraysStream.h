#ifndef _DARRAYSSTREAM_H_
#define _DARRAYSSTREAM_H_

#include "DIOStream.h"

namespace smil {
    template <class T>
    class ArraysStream : public IOStream <T>{
        public:
            typedef IOStream<T> parentClass;

            ArraysStream () : r_head (0) {}
            RES_T initialize (const int nbr_procs, const int _intersect_width, Image<T>& imI, Image<T>& imO) {
                size_t s[3];
                imI.getSize(s);
                int _s[3];
                for (int i=0; i<3; ++i) _s[i] = s[i];
                int _mpi_datum_type = smilToMPIType (imI.getTypeAsString());
                
                return initialize (nbr_procs, _intersect_width, _mpi_datum_type, _s, imI.getPixels(), imO.getPixels()); 
            }
            RES_T initialize (const int nbr_procs, const int _intersect_width, const int _mpi_datum_type, int* _size, T* _dataI, T* _dataO) {
                size = new int[3];
                memcpy (size, _size, 3*sizeof(int)) ;
                dataI = _dataI;
                dataO = _dataO;
                parentClass::mpi_datum_type = _mpi_datum_type;
                parentClass::nbr_diff_chunks = 3;
                intersect_width = _intersect_width;

                parentClass::chunks_per_dims[0] = 1;
                parentClass::chunks_per_dims[1] = 1;
                parentClass::chunks_per_dims[2] = 1;
                
                while (parentClass::chunks_per_dims[0]*parentClass::chunks_per_dims[1]*parentClass::chunks_per_dims[2] < nbr_procs) {
                    if (size[0]/parentClass::chunks_per_dims[0] >= size[1]/parentClass::chunks_per_dims[1] && size[0]/parentClass::chunks_per_dims[0] >= size[2]/parentClass::chunks_per_dims[2]) {
                        parentClass::chunks_per_dims[0]++;
                    } else if (size[1]/parentClass::chunks_per_dims[1] >= size[2]/parentClass::chunks_per_dims[2]) {
                        parentClass::chunks_per_dims[1]++;
                    } else {
                        parentClass::chunks_per_dims[2]++;
                    }
                }

                cout << parentClass::chunks_per_dims[0] <<  " " << parentClass::chunks_per_dims[1] << " " << parentClass::chunks_per_dims[2] << endl;

                parentClass::chunks_nbr = new int[3];
                parentClass::chunks_len = new int[3];     
                parentClass::chunks_nbr[1] =      parentClass::chunks_per_dims[0]*parentClass::chunks_per_dims[1] +
                                     parentClass::chunks_per_dims[0]*parentClass::chunks_per_dims[2] + 
                                     parentClass::chunks_per_dims[1]*parentClass::chunks_per_dims[2] -
                                     parentClass::chunks_per_dims[0]-parentClass::chunks_per_dims[1] -
                                     parentClass::chunks_per_dims[2] +1;
                parentClass::chunks_len[1] =  (size[0] - size[0]/parentClass::chunks_per_dims[0] * (parentClass::chunks_per_dims[0]-1)) *
                                    (size[1] - size[1]/parentClass::chunks_per_dims[1] * (parentClass::chunks_per_dims[1]-1)) *
                                    (size[2] - size[2]/parentClass::chunks_per_dims[2] * (parentClass::chunks_per_dims[2]-1));
                parentClass::chunks_nbr[0] = parentClass::chunks_per_dims[0] * parentClass::chunks_per_dims[1] * parentClass::chunks_per_dims[2] - parentClass::chunks_nbr[1];
                parentClass::chunks_len[0] = size[0]/parentClass::chunks_per_dims[0] * size[1]/parentClass::chunks_per_dims[1] * size[2]/parentClass::chunks_per_dims[2];
                parentClass::chunks_nbr[2] = parentClass::chunks_per_dims[0] + parentClass::chunks_per_dims[1] + parentClass::chunks_per_dims[2] -3;
                int max_dimension = MAX(size[1],size[2]);
                max_dimension = MAX(max_dimension,size[0]);
                parentClass::chunks_len[2] = (_intersect_width*2+1) * max_dimension * max_dimension; 

                parentClass::nbr_chunks = parentClass::chunks_nbr[0] + parentClass::chunks_nbr[1] + parentClass::chunks_nbr[2];
                cout << "-----> " << parentClass::nbr_chunks << endl;

                return RES_OK;
            }
            RES_T read_normal (const int pos, Chunk<T> &c) {
                int i,j,k,s_x,s_y,s_z;
                k = pos/(parentClass::chunks_per_dims[0]*parentClass::chunks_per_dims[1]); 
                j = (pos-k*parentClass::chunks_per_dims[1])/(parentClass::chunks_per_dims[0]); 
                i = pos-j*parentClass::chunks_per_dims[0]*parentClass::chunks_per_dims[1]; 
                s_x = (i == parentClass::chunks_per_dims[0]-1) ? size[0] - i*size[0]/parentClass::chunks_per_dims[0] : size[0]/parentClass::chunks_per_dims[0];
                i *= size[0]/parentClass::chunks_per_dims[0];
                s_y = (j == parentClass::chunks_per_dims[1]-1) ? size[1] - j*size[1]/parentClass::chunks_per_dims[1] : size[1]/parentClass::chunks_per_dims[1];
                j *= size[1]/parentClass::chunks_per_dims[1];
                s_z = (k == parentClass::chunks_per_dims[2]-1) ? size[2] - k*size[2]/parentClass::chunks_per_dims[2] : size[2]/parentClass::chunks_per_dims[2];
                k *= size[2]/parentClass::chunks_per_dims[2];

                c.createFromArray (
                   dataI,
                   size[0], size[1],
                   i,
                   j,
                   k,
                   s_x,
                   s_y,
                   s_z
                   );
                c.print () ;
                return RES_OK;
            }
            RES_T read_intersect (const int pos, Chunk<T> &c) {
                int i;
                if (pos < parentClass::chunks_per_dims[0]-1) {
                    i = (pos+1)*size[0]/parentClass::chunks_per_dims[0]-intersect_width;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            i,
                            0,
                            0,
                            intersect_width*2+1,
                            size[1],
                            size[2] 
                            );
                } else if (pos-parentClass::chunks_per_dims[0]+1 < parentClass::chunks_per_dims[1]-1) {
                    i = (pos-parentClass::chunks_per_dims[0]+2)*size[1]/parentClass::chunks_per_dims[1]-intersect_width;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            0,
                            i,
                            0,
                            size[0],
                            intersect_width*2+1,
                            size[2] 
                            );               
                } else {
                    i = (pos-parentClass::chunks_per_dims[0]-parentClass::chunks_per_dims[1]+3)*size[2]/parentClass::chunks_per_dims[2]-intersect_width;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            0,
                            0,
                            i,
                            size[0],
                            size[1], 
                            intersect_width*2+1
                            );
                }
                c.print () ;
                return RES_OK;
            }
            RES_T next (Chunk<T> &c) {
                cout << r_head << endl;
                if (r_head < parentClass::chunks_nbr[1]+parentClass::chunks_nbr[0]) 
                    return read_normal (r_head++, c);
                return read_intersect ((r_head++ - parentClass::chunks_nbr[1] - parentClass::chunks_nbr[0]), c);
            }
            RES_T recv () {
                // mpi_recv
                // store in dataO
            }
            bool eof () {
                return r_head == parentClass::nbr_chunks;
            } 
        private:
            int intersect_width;
            int r_head;
            int* size;
            T* dataI;
            T* dataO;
    };
}

#endif
