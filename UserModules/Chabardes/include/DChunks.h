#ifndef _DCHUNKS_H_
#define _DCHUNKS_H_

#include "DImage.h"

namespace smil {
    template <class T>
    struct Chunk {
        int size[3];
        int offset[3];
        T* data;
    };

    template <class T>
    void printChunk (Chunk<T> c) {
        cout << endl << "#### size : [" << c.size[0] << ", " << c.size[1] << ", " << c.size[2] << "], offset : [" << c.offset[0] << ", " << c.offset[1] << ", " << c.offset[2] << "] ####" << endl;
        for (int i=0; i<c.size[2]; ++i) {
            for (int j=0; j<c.size[1]; ++j) {
                for (int k=0; k<c.size[0]; ++k)
                    cout << (int) c.data[k+j*c.size[0]+i*c.size[0]*c.size[1]] << ' ';
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

    template <class T>
    RES_T copyChunkFromArray (const T* dataIn, const int o_x, const int o_y, const int o_z, const int s_x, const int s_y, const int s_z, const int sd_x, const int sd_y, Chunk<T>& c) {
//        cout << "offset : [" << o_x << ", " << o_y << ", " << o_z << "], size : [" << s_x << ", " << s_y << ", " << s_z << "]." << endl;

        c.size[0] = s_x; c.size[1] = s_y; c.size[2] = s_z;
        c.offset[0] = o_x; c.offset[1] = o_y; c.offset[2] = o_z; 
        c.data = new T[s_x*s_y*s_z];

        for (int k=0; k<s_z; ++k) {
            for (int j=0; j<s_y; ++j) {
                if (memcpy (c.data + j * s_x + k * s_x * s_y, 
                            dataIn + o_x + (o_y+j) * sd_x + (o_z+k) * sd_x * sd_y,
                            s_x*sizeof(T)) == NULL){
                    return RES_ERR_IO;
                }
            }
        }
        return RES_OK;
    }

    template <class T>
    RES_T getNumberOfChunksIntersects (const Image<T> &imIn, int line_size, int chunk_size, int min_nbr_chunks, int& nbr_chunks, int& nbr_intersects, int* ncpd) {
        ncpd[0] = 1; ncpd[1] = 1; ncpd[2] = 1; 
        size_t s[3]; 
        imIn.getSize(s);

        while (ncpd[0]*ncpd[1]*ncpd[2] < min_nbr_chunks) {
            if (s[0]/ncpd[0] >= s[1]/ncpd[1] && s[0]/ncpd[0] >= s[2]/ncpd[2]) {
                ncpd[0]++;
            } else if (s[1]/ncpd[1] >= s[2]/ncpd[2]) {
                ncpd[1]++;
            } else {
                ncpd[2]++;
            }
        }
  
        nbr_chunks = ncpd[0]*ncpd[1]*ncpd[2];
        nbr_intersects = ncpd[0] + ncpd[1] + ncpd[2] -3;

//        cout << "number of chunks : " << nbr_chunks << " [" << ncpd[0] << ", " << ncpd[1] << ", " << ncpd[2] << "], number of intersects : " << nbr_intersects << endl; 
    }

    template <class T>
    RES_T createChunks (const Image<T> &imIn, int nbr_chunks, int* ncpd, Chunk<T>* &ca) {
        RES_T err;
        size_t s[3]; 
        imIn.getSize(s);

        ca = new Chunk<T>[ncpd[0]*ncpd[1]*ncpd[2]];

        int s_x, s_y, s_z;
        for (int k=0; k<ncpd[2]; ++k) {
            for (int j=0; j<ncpd[1]; ++j) {
                for (int i=0; i<ncpd[0]; ++i) {
                    s_x = (i == ncpd[0]-1 ) ? s[0] - (s[0]/ncpd[0]) * i : s[0]/ncpd[0] ;
                    s_y = (j == ncpd[1]-1 ) ? s[1] - (s[1]/ncpd[1]) * j : s[1]/ncpd[1] ;
                    s_z = (k == ncpd[2]-1 ) ? s[2] - (s[2]/ncpd[2]) * k : s[2]/ncpd[2] ;

                    err = copyChunkFromArray ((T*)imIn.getPixels(),
                                        i*s[0]/ncpd[0], 
                                        j*s[1]/ncpd[1], 
                                        k*s[2]/ncpd[2],
                                        s_x,
                                        s_y,
                                        s_z,
                                        s[0], s[1], 
                                        ca[i+j*ncpd[0]+k*ncpd[0]*ncpd[1]]
                                       );
                    if (err != RES_OK)
                        return RES_ERR_BAD_ALLOCATION;
                }
            }
        }
        return RES_OK;
    }

    template <class T>
    RES_T createIntersects (const Image<T> &imIn, int ncpd[], int nbr_intersects, int width, Chunk<T>* &ia) {
        RES_T err;
        size_t s[3];
        imIn.getSize(s);

        ia = new Chunk<T>[nbr_intersects];
        int j=0;

        for (int i=1; i<ncpd[0]; ++i, ++j) {
            err = copyChunkFromArray (
                                     (T*)imIn.getPixels(),
                                      i*s[0]/ncpd[0]-width/2,
                                      0,
                                      0,
                                      width,
                                      s[1],
                                      s[2],
                                      s[0],
                                      s[1],
                                      ia[j]
                                     );
        }
        for (int i=1; i<ncpd[1]; ++i, ++j) {
            err = copyChunkFromArray (
                                     (T*)imIn.getPixels(), 
                                      0,
                                      i*s[1]/ncpd[1]-width/2,
                                      0,
                                      s[0],
                                      width,
                                      s[2],
                                      s[0],
                                      s[1],
                                      ia[j]
                                     );
        }
        for (int i=1; i<ncpd[2]; ++i, ++j) {
            err = copyChunkFromArray (
                                      (T*)imIn.getPixels(), 
                                      0,
                                      0,
                                      i*s[2]/ncpd[2]-width/2,
                                      s[0],
                                      s[1],
                                      width,
                                      s[0],
                                      s[1],
                                      ia[j]
                                     );
        }
        return RES_OK;
    }

    template <class T>
    RES_T storeChunks (Chunk<T>* ca, Image<T> &imIn, int nbr_chunks) {
    
    }

    template <class T>
    RES_T deleteChunks (Chunk<T>* ca, int nbr_chunks) {
    
    }
}

#endif
