#ifndef _DARRAYSSTREAM_H_
#define _DARRAYSSTREAM_H_

#include "DIOStream.h"

namespace smil {
    template <class T>
    class Arrays_Stream : public IOStream {
        public:
            Arrays_Stream () : r_head (0) {}
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
                mpi_datum_type = _mpi_datum_type;
                nbr_diff_chunks = 3;
                intersect_width = _intersect_width;

                chunks_per_dims[0] = 1;
                chunks_per_dims[1] = 1;
                chunks_per_dims[2] = 1;
                
                while (chunks_per_dims[0]*chunks_per_dims[1]*chunks_per_dims[2] < nbr_procs) {
                    if (size[0]/chunks_per_dims[0] >= size[1]/chunks_per_dims[1] && size[0]/chunks_per_dims[0] >= size[2]/chunks_per_dims[2]) {
                        chunks_per_dims[0]++;
                    } else if (size[1]/chunks_per_dims[1] >= size[2]/chunks_per_dims[2]) {
                        chunks_per_dims[1]++;
                    } else {
                        chunks_per_dims[2]++;
                    }
                }

                cout << chunks_per_dims[0] <<  " " << chunks_per_dims[1] << " " << chunks_per_dims[2] << endl;

                chunks_nbr = new int[3];
                chunks_len = new int[3];     
                chunks_nbr[1] =      chunks_per_dims[0]*chunks_per_dims[1] +
                                     chunks_per_dims[0]*chunks_per_dims[2] + 
                                     chunks_per_dims[1]*chunks_per_dims[2] -
                                     chunks_per_dims[0]-chunks_per_dims[1] -
                                     chunks_per_dims[2] +1;
                chunks_len[1] =  (size[0] - size[0]/chunks_per_dims[0] * (chunks_per_dims[0]-1)) *
                                    (size[1] - size[1]/chunks_per_dims[1] * (chunks_per_dims[1]-1)) *
                                    (size[2] - size[2]/chunks_per_dims[2] * (chunks_per_dims[2]-1));
                chunks_nbr[0] = chunks_per_dims[0] * chunks_per_dims[1] * chunks_per_dims[2] - chunks_nbr[1];
                chunks_len[0] = size[0]/chunks_per_dims[0] * size[1]/chunks_per_dims[1] * size[2]/chunks_per_dims[2];
                chunks_nbr[2] = chunks_per_dims[0] + chunks_per_dims[1] + chunks_per_dims[2] -3;
                int max_dimension = MAX(size[1],size[2]);
                max_dimension = MAX(max_dimension,size[0]);
                chunks_len[2] = (_intersect_width*2+1) * max_dimension * max_dimension; 

                nbr_chunks = chunks_nbr[0] + chunks_nbr[1] + chunks_nbr[2];
                cout << "-----> " << nbr_chunks << endl;

                return RES_OK;
            }
            RES_T read_normal (const int pos, Chunk<T> &c) {
                int i,j,k,s_x,s_y,s_z;
                k = pos/(chunks_per_dims[0]*chunks_per_dims[1]); 
                j = (pos-k*chunks_per_dims[1])/(chunks_per_dims[0]); 
                i = pos-j*chunks_per_dims[0]*chunks_per_dims[1]; 
                s_x = size[0]/chunks_per_dims[0];
                i *= s_x;
                s_x -= i;
                s_y = size[1]/chunks_per_dims[1];
                j *= s_y;
                s_y -= j;
                s_z = size[2]/chunks_per_dims[2];
                k *= s_z;
                s_z -= k;

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
                return RES_OK;
            }
            RES_T read_intersect (const int pos, Chunk<T> &c) {
                int i;
                if (pos < chunks_per_dims[0]-1) {
                    i = (pos+1)*size[0]/chunks_per_dims[0]-intersect_width/2;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            i,
                            0,
                            0,
                            intersect_width,
                            size[1],
                            size[2] 
                            );
                } else if (pos-chunks_per_dims[0]+1 < chunks_per_dims[1]-1) {
                    i = (pos-chunks_per_dims[0]+1)*size[1]/chunks_per_dims[1]-intersect_width/2;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            0,
                            i,
                            0,
                            size[0],
                            intersect_width,
                            size[2] 
                            );               
                } else {
                    i = (pos-chunks_per_dims[0]-chunks_per_dims[1]+2)*size[2]/chunks_per_dims[2]-intersect_width/2;
                    c.createFromArray (
                            dataI,
                            size[0], size[1],
                            0,
                            0,
                            i,
                            size[0],
                            size[1], 
                            intersect_width
                            );
                }
                return RES_OK;
            }
            RES_T next (Chunk<T> &c) {
                cout << r_head << endl;
                if (r_head < chunks_nbr[1]+chunks_nbr[0]) 
                    return read_normal (r_head++, c);
                return read_intersect ((r_head++ - chunks_nbr[1] - chunks_nbr[0]), c);
            }
            RES_T recv () {
                // mpi_recv
                // store in dataO
            }
            bool eof () {
                return r_head == nbr_chunks;
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
