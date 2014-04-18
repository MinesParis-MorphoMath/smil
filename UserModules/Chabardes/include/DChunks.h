#ifndef _DCHUNKS_H_
#define _DCHUNKS_H_

#include "DImage.h"
#include "DMorpho.h"
#include "mpi.h"

namespace smil {
    template <class T>
    struct Chunk {
        int size[3];
        int offset[3];
        T* data;
    };

    struct Chunks_Header {
        int datum_size;
        int mpi_type;
        int nbr_chunks;
        int size_chunks;
        int nbr_borders;
        int size_borders;
        int nbr_intersects;
        int size_intersects;
        int ncpd[3]; // number of chunks per dimensions.
    };

    template <class T>
    struct Chunks_Data {
        Chunk<UINT8>* ca; // chunks
        Chunk<UINT8>* ba; // borders
        Chunk<UINT8>* ia; // intersections  
    };

    int smilToMPIType (const char* type_datum) {
        if (type_datum == "UINT8") {
                return MPI_UNSIGNED_CHAR;
        } else if (type_datum == "UINT16") {
                return MPI_UNSIGNED_SHORT;
        } else if (type_datum == "UINT32") {
                return MPI_UNSIGNED;
        } else if (type_datum == "UINT64") {
                return MPI_UNSIGNED_LONG;
        } else if (type_datum == "INT") {
                return MPI_INT;
        } else {
                return MPI_UNSIGNED;
        }
    }

    template <class T>
    void printChunk (Chunk<T> c, bool print_data=false) {
        cout << "#### size : [" << c.size[0] << ", " << c.size[1] << ", " << c.size[2] << "], offset : [" << c.offset[0] << ", " << c.offset[1] << ", " << c.offset[2] << "] ####" << endl;
        if (!print_data) return;
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
    RES_T getChunksHeader (Image<T> &imIn, const int size_simultaneously_calculated, const int size_chunk_max, const int nbr_cores, const StrElt &se, const int intersect_width, Chunks_Header &ch) {
        ch.ncpd[0] = 1; ch.ncpd[1] = 1; ch.ncpd[2] = 1; 
        size_t s[3]; 
        imIn.getSize(s);

        ch.mpi_type = smilToMPIType (imIn.getTypeAsString()) ;
        ch.datum_size = sizeof (T); 

        while (ch.ncpd[0]*ch.ncpd[1]*ch.ncpd[2] < nbr_cores) {
            if (s[0]/ch.ncpd[0] >= s[1]/ch.ncpd[1] && s[0]/ch.ncpd[0] >= s[2]/ch.ncpd[2]) {
                ch.ncpd[0]++;
            } else if (s[1]/ch.ncpd[1] >= s[2]/ch.ncpd[2]) {
                ch.ncpd[1]++;
            } else {
                ch.ncpd[2]++;
            }
        }
  
        ch.nbr_borders = ch.ncpd[0]*ch.ncpd[1] + ch.ncpd[0]*ch.ncpd[2] + ch.ncpd[1]*ch.ncpd[2] -ch.ncpd[0] -ch.ncpd[1] -ch.ncpd[2] +1;
        ch.size_borders =  (s[0 - s[0]/ch.ncpd[0]] * ch.ncpd[0]-1) *
                            (s[1 - s[1]/ch.ncpd[1]] * ch.ncpd[1]-1) *
                            (s[2 - s[2]/ch.ncpd[2]] * ch.ncpd[2]-1);
        ch.nbr_chunks = ch.ncpd[0]*ch.ncpd[1]*ch.ncpd[2] - ch.nbr_borders;
        ch.size_chunks = s[0]/ch.ncpd[0] * s[1]/ch.ncpd[1] * s[2]/ch.ncpd[2];
        ch.nbr_intersects = ch.ncpd[0] + ch.ncpd[1] + ch.ncpd[2] -3;
        int max_dimension = MAX(s[1],s[2]);
        max_dimension = MAX(max_dimension,s[0]);
        ch.size_intersects = intersect_width * max_dimension*max_dimension; 

//        cout << "number of chunks : " << ch.nbr_chunks+ch.nbr_borders << " [" << ch.ncpd[0] << ", " << ch.ncpd[1] << ", " << ch.ncpd[2] << "], number of intersects : " << ch.nbr_intersects << endl; 
    }

    template <class T>
    RES_T createChunks (const Image<T> &imIn, const Chunks_Header &ch, Chunks_Data<T> &cd) {
        RES_T err;
        size_t s[3]; 
        imIn.getSize(s);

        cd.ca = new Chunk<T>[ch.nbr_chunks];
        cd.ba = new Chunk<T>[ch.nbr_borders];
        Chunk<T>* target;
        int v=0, w=0;

        int s_x, s_y, s_z;
        for (int k=0; k<ch.ncpd[2]; ++k) {
            for (int j=0; j<ch.ncpd[1]; ++j) {
                for (int i=0; i<ch.ncpd[0]; ++i) {
                    s_x = (i == ch.ncpd[0]-1 ) ? s[0] - (s[0]/ch.ncpd[0]) * i : s[0]/ch.ncpd[0] ;
                    s_y = (j == ch.ncpd[1]-1 ) ? s[1] - (s[1]/ch.ncpd[1]) * j : s[1]/ch.ncpd[1] ;
                    s_z = (k == ch.ncpd[2]-1 ) ? s[2] - (s[2]/ch.ncpd[2]) * k : s[2]/ch.ncpd[2] ;
                    if (i == ch.ncpd[0]-1 || j == ch.ncpd[1]-1 || k == ch.ncpd[2]-1) {target = cd.ba + (v++);}
                    else {target = cd.ca + (w++);}

                    err = copyChunkFromArray ((T*)imIn.getPixels(),
                                        i*s[0]/ch.ncpd[0], 
                                        j*s[1]/ch.ncpd[1], 
                                        k*s[2]/ch.ncpd[2],
                                        s_x,
                                        s_y,
                                        s_z,
                                        s[0], s[1], 
                                        *target
                                       );
                    if (err != RES_OK)
                        return RES_ERR_BAD_ALLOCATION;
                }
            }
        }
        return RES_OK;
    }

    template <class T>
    RES_T createIntersects (const Image<T> &imIn, const int& intersect_width, const Chunks_Header &ch, Chunks_Data<T> &cd) {
        RES_T err;
        size_t s[3];
        imIn.getSize(s);

        cd.ia = new Chunk<T>[ch.nbr_intersects];
        int j=0;

        for (int i=1; i<ch.ncpd[0]; ++i, ++j) {
            err = copyChunkFromArray (
                                     (T*)imIn.getPixels(),
                                      i*s[0]/ch.ncpd[0]-intersect_width/2,
                                      0,
                                      0,
                                      intersect_width,
                                      s[1],
                                      s[2],
                                      s[0],
                                      s[1],
                                      cd.ia[j]
                                     );
        }
        for (int i=1; i<ch.ncpd[1]; ++i, ++j) {
            err = copyChunkFromArray (
                                     (T*)imIn.getPixels(), 
                                      0,
                                      i*s[1]/ch.ncpd[1]-intersect_width/2,
                                      0,
                                      s[0],
                                      intersect_width,
                                      s[2],
                                      s[0],
                                      s[1],
                                      cd.ia[j]
                                     );
        }
        for (int i=1; i<ch.ncpd[2]; ++i, ++j) {
            err = copyChunkFromArray (
                                      (T*)imIn.getPixels(), 
                                      0,
                                      0,
                                      i*s[2]/ch.ncpd[2]-intersect_width/2,
                                      s[0],
                                      s[1],
                                      intersect_width,
                                      s[0],
                                      s[1],
                                      cd.ia[j]
                                     );
        }
        return RES_OK;
    }

// TODO: check why imIn has to be not const to use getTypeAsString () .
    template <class T>
    RES_T generateChunks (Image<T> &imIn, const int size_simultaneously_calculated, const int size_chunk_max, const int nbr_cores, const StrElt &se, const int intersect_width, Chunks_Header &ch, Chunks_Data<T> &cd ) {
        getChunksHeader (imIn, size_simultaneously_calculated, size_chunk_max, nbr_cores, se, intersect_width, ch);
        createChunks (imIn, ch, cd);
        createIntersects (imIn, intersect_width, ch, cd);
    }



    int registerChunks (const Chunks_Header &ch, MPI_Datatype &mpi_chunk_type, MPI_Datatype &mpi_border_type, MPI_Datatype &mpi_intersect_type) {
//    Broadcast the infos on the chunks.
        MPI_Bcast ((void*)&ch, 11, MPI_INTEGER, 0, MPI_COMM_WORLD);

        MPI_Datatype types[2] = {MPI_INTEGER, ch.mpi_type};
        int blocks_lengths[2] = {6,ch.nbr_chunks};
        MPI_Aint offsets[2] = {0, 6*sizeof(int)} ;

//    CHUNKS
        MPI_Type_create_struct (2, blocks_lengths, offsets, types, &mpi_chunk_type) ;
        MPI_Type_commit (&mpi_chunk_type);
//    BORDERS
        blocks_lengths[1] = ch.nbr_borders;
        MPI_Type_create_struct (2, blocks_lengths, offsets, types, &mpi_border_type) ;
        MPI_Type_commit (&mpi_chunk_type);
//    INTERSECTS
        blocks_lengths[1] = ch.nbr_intersects;
        MPI_Type_create_struct (2, blocks_lengths, offsets, types, &mpi_intersect_type) ;
        MPI_Type_commit (&mpi_chunk_type);
    }

    RES_T unregisterChunks (const Chunks_Header &ch, MPI_Datatype &mpi_chunk_type, MPI_Datatype &mpi_border_type, MPI_Datatype &mpi_intersect_type) {
        MPI_Type_free (&mpi_chunk_type);
        MPI_Type_free (&mpi_border_type);
        MPI_Type_free (&mpi_intersect_type);
    }

    template <class T>
    RES_T deleteChunks (const Chunks_Header &ch, Chunks_Data<T> &cd) {
        for (int i=0; i<ch.nbr_chunks; ++i) {
            delete cd.ca[i].data;
        }
        delete cd.ca;
        for (int i=0; i<ch.nbr_borders; ++i) {
            delete cd.ba[i].data;
        }
        delete cd.ba;
        for (int i=0; i<ch.nbr_intersects; ++i) {
            delete cd.ia[i].data;
        }
        delete cd.ia;
    }
}

#endif
