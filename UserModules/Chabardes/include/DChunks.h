#ifndef _DCHUNKS_H_
#define _DCHUNKS_H_

#include "DImage.h"
#include "DMorpho.h"
#include "mpi.h"


//TODO: supprimer intersect_width et remplacer avec le rayon de l'element structurant.

namespace smil {

    template <class T>
    class Chunk {
        public:
            Chunk (): rawData(NULL)  { }
            ~Chunk () {
                cout << "deleting " << hex << (long) rawData << " ..." << endl;
                if (rawData != NULL) 
                    ::operator delete(rawData);
            }
            RES_T allocate (const size_t &s) {
                rawData = ::operator new (sizeof(int)*6+sizeof(T)*s);

                cout << "allocating " << hex << (long) rawData << " ..." << endl;
                if (rawData == NULL) { return RES_ERR_BAD_ALLOCATION; } 
                size = (int*)rawData;
                offset = (int*)rawData + 3;
                data = (T*)((int*)rawData + 6);
            }
            RES_T createFromArray (
                    const T* dataIn,
                    const int &sd_x, 
                    const int &sd_y,
                    const int &o_x,
                    const int &o_y,
                    const int &o_z,
                    const int &sc_x,
                    const int &sc_y,
                    const int &sc_z) {
                size[0] = sc_x; size[1] = sc_y; size[2] = sc_z;
                offset[0] = o_x; offset[1] = o_y; offset[2] = o_z; 

                for (int k=0; k<sc_z; ++k) {
                    for (int j=0; j<sc_y; ++j) {
                        if (memcpy (data + j * sc_x + k * sc_x * sc_y, 
                                    dataIn + o_x + (o_y+j) * sd_x + (o_z+k) * sd_x * sd_y,
                                    sc_x*sizeof(T)) == NULL){
                            return RES_ERR_IO;
                        }
                    }
                }
                return RES_OK;
                   
            }
            RES_T storeToArray (const int &sd_x, const int &sd_y, T* dataOut) {
                for (int k=0; k<size[2]; ++k) {
                    for (int j=0; j<size[1]; ++j) {
                        if (memcpy (dataOut + offset[0] + (offset[1]+j) * sd_x + (offset[2]+k) * sd_x * sd_y,
                                    data + j * size[0] + k * size[0] * size[1], 
                                    size[0]*sizeof(T)) == NULL){
                            return RES_ERR_IO;
                        }
                    }
                }
                return RES_OK;
            } 
            RES_T storeToImage (Image<T> &imOut) {
//              ASSERT (all the good stuff).

                size_t s[3];
                imOut.getSize(s);

                if (storeToArray (s[0], s[1], imOut.getPixels()) != RES_OK) return RES_ERR;
                return RES_OK;
            }
            int getSize (const unsigned int &dimension) {
                ASSERT (dimension < 3) ;
                return size[dimension] ;
            }
            int getOffset (const unsigned int &dimension) {
                ASSERT (dimension < 3) ;
                return offset[dimension] ;

            }
            T* getData () {
                return data;
            }
            void print (bool print_data = false) {
                ASSERT (rawData!=NULL) ;
                cout << "#### size : [" << size[0] << ", " << size[1] << ", " << size[2] << "], offset : [" << offset[0] << ", " << offset[1] << ", " << offset[2] << "] ####" << endl;
                if (!print_data) return;
                for (int i=0; i<size[2]; ++i) {
                    for (int j=0; j<size[1]; ++j) {
                        for (int k=0; k<size[0]; ++k)
                            cout << (int) data[k+j*size[0]+i*size[0]*size[1]] << ' ';
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl; 
            }
        private:
            void* rawData;
            int* size;
            int* offset;
            T* data;
    };

    struct Chunks_Header {
        int datum_size;
        int nbr_chunks;
        int size_chunks;
        int nbr_borders;
        int size_borders;
        int nbr_intersects;
        int size_intersects;
        int ncpd[3]; // number of chunks per dimensions.
        int mpi_type;
        MPI_Datatype mpi_chunks_types[3];
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
        ch.size_borders =  (s[0] - s[0]/ch.ncpd[0] * (ch.ncpd[0]-1)) *
                            (s[1] - s[1]/ch.ncpd[1] * (ch.ncpd[1]-1)) *
                            (s[2] - s[2]/ch.ncpd[2] * (ch.ncpd[2]-1));
        ch.nbr_chunks = ch.ncpd[0]*ch.ncpd[1]*ch.ncpd[2] - ch.nbr_borders;
        ch.size_chunks = s[0]/ch.ncpd[0] * s[1]/ch.ncpd[1] * s[2]/ch.ncpd[2];
        ch.nbr_intersects = ch.ncpd[0] + ch.ncpd[1] + ch.ncpd[2] -3;
        int max_dimension = MAX(s[1],s[2]);
        max_dimension = MAX(max_dimension,s[0]);
        ch.size_intersects = intersect_width * max_dimension*max_dimension; 
    }

    template <class T>
    RES_T createChunks (const Image<T> &imIn, const Chunks_Header &ch, Chunk<T> *ca) {
        RES_T err;

        for (int i=0; i<ch.nbr_chunks; ++i) {
            ca[i].allocate (ch.size_chunks);
        }
        for (int i=0;i<ch.nbr_borders; ++i) {
            ca[ch.nbr_chunks+i].allocate (ch.size_borders);
        }

        size_t s[3]; 
        imIn.getSize(s);

        int target; 
        int v=0, w=0;

        int s_x, s_y, s_z;
        for (int k=0; k<ch.ncpd[2]; ++k) {
            for (int j=0; j<ch.ncpd[1]; ++j) {
                for (int i=0; i<ch.ncpd[0]; ++i) {
                    s_x = (i == ch.ncpd[0]-1 ) ? s[0] - (s[0]/ch.ncpd[0]) * i : s[0]/ch.ncpd[0] ;
                    s_y = (j == ch.ncpd[1]-1 ) ? s[1] - (s[1]/ch.ncpd[1]) * j : s[1]/ch.ncpd[1] ;
                    s_z = (k == ch.ncpd[2]-1 ) ? s[2] - (s[2]/ch.ncpd[2]) * k : s[2]/ch.ncpd[2] ;
                    if (i == ch.ncpd[0]-1 || j == ch.ncpd[1]-1 || k == ch.ncpd[2]-1) {target = ch.nbr_chunks + (v++);}
                    else {target = w++;}

                    if (ca[target].createFromArray (
                                        (T*)imIn.getPixels(),
                                        s[0], s[1], 
                                        i*s[0]/ch.ncpd[0], 
                                        j*s[1]/ch.ncpd[1], 
                                        k*s[2]/ch.ncpd[2],
                                        s_x,
                                        s_y,
                                        s_z
                                       ) != RES_OK)
                        return RES_ERR_BAD_ALLOCATION;
                }
            }
        }
        return RES_OK;
    }

    template <class T>
    RES_T createIntersects (const Image<T> &imIn, const int& intersect_width, const Chunks_Header &ch, Chunk<T>* ca) {
        RES_T err;

        for (int i=0; i<ch.nbr_intersects; ++i) {
            ca[ch.nbr_chunks+ch.nbr_borders+i].allocate (ch.size_intersects) ;
        }

        size_t s[3];
        imIn.getSize(s);

        int j=0;

        for (int i=1; i<ch.ncpd[0]; ++i, ++j) {
            ca[ch.nbr_chunks+ch.nbr_borders+j].createFromArray (
                                     (T*)imIn.getPixels(),
                                      s[0], s[1],
                                      i*s[0]/ch.ncpd[0]-intersect_width/2,
                                      0,
                                      0,
                                      intersect_width,
                                      s[1],
                                      s[2]
                                     );
        }
        for (int i=1; i<ch.ncpd[1]; ++i, ++j) {
            ca[ch.nbr_chunks+ch.nbr_borders+j].createFromArray (
                                     (T*)imIn.getPixels(), 
                                      s[0], s[1],
                                      0,
                                      i*s[1]/ch.ncpd[1]-intersect_width/2,
                                      0,
                                      s[0],
                                      intersect_width,
                                      s[2]
                                     );
        }
        for (int i=1; i<ch.ncpd[2]; ++i, ++j) {
            ca[ch.nbr_chunks+ch.nbr_borders+j].createFromArray (
                                      (T*)imIn.getPixels(), 
                                      s[0], s[1],
                                      0,
                                      0,
                                      i*s[2]/ch.ncpd[2]-intersect_width/2,
                                      s[0],
                                      s[1],
                                      intersect_width
                                     );
        }
        return RES_OK;
    }

// TODO: check why imIn has to be not const to use getTypeAsString () .
    template <class T>
    RES_T generateChunks (Image<T> &imIn, const int size_simultaneously_calculated, const int size_chunk_max, const int nbr_cores, const StrElt &se, const int intersect_width, Chunks_Header &ch, Chunk<T>* &ca ) {
        getChunksHeader (imIn, size_simultaneously_calculated, size_chunk_max, nbr_cores, se, intersect_width, ch);
        ca = new Chunk<T>[ch.nbr_chunks+ch.nbr_borders+ch.nbr_intersects];
        createChunks (imIn, ch, ca);
        createIntersects (imIn, intersect_width, ch, ca);
    }

    void registerChunks (Chunks_Header &ch) { 
//    Broadcast the infos on the chunks.
        MPI_Bcast ((void*)&ch, 11, MPI_INTEGER, 0, MPI_COMM_WORLD);
        MPI_Datatype old_types[2] = {MPI_INTEGER, ch.mpi_type};
        MPI_Aint steps[2] = {0,6*sizeof(int)};
        int blocks_sizes [2]; blocks_sizes[0] = 6; 
 
//    CHUNKS 
        blocks_sizes[1] = ch.size_chunks;
        MPI_Type_struct (2, blocks_sizes, steps, old_types, ch.mpi_chunks_types);
//    BORDERS
        blocks_sizes[1] = ch.size_borders;
        MPI_Type_struct (2, blocks_sizes, steps, old_types, ch.mpi_chunks_types+1);
//    INTERSECTS
        blocks_sizes[2] = ch.size_intersects;
        MPI_Type_struct (2, blocks_sizes, steps, old_types, ch.mpi_chunks_types+2);
    }

    void unregisterChunks (Chunks_Header &ch) {
        for (int i=0; i<3; ++i) {
            MPI_Type_free (ch.mpi_chunks_types+i) ;
        }
    }
}

#endif
