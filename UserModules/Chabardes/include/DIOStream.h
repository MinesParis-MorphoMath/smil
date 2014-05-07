#ifndef _DIOSTREAM_H_
#define _DIOSTREAM_H_

#include <DChunk.h>

namespace smil {
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
    class IOStream {
        protected:
            // Need initializing.
            int nbr_diff_chunks;
            int nbr_chunks;
            int mpi_datum_type;
            int *chunks_len;
            int *chunks_nbr;
            int chunks_per_dims[3];
            // everything else.
            MPI_Datatype *mpi_types;
        public:
            bool eof () { 
                return false; 
            } 
#define MAX_DIFF_CHUNKS 10
            RES_T mpiRegisterMaster () {
                int *mpi_packet = new int[6+MAX_DIFF_CHUNKS*2];
                mpi_packet[0] = nbr_diff_chunks;
                mpi_packet[1] = nbr_chunks;
                mpi_packet[2] = mpi_datum_type;
                for (int i=0; i<nbr_diff_chunks; ++i) mpi_packet[i+3] = chunks_len[i];
                for (int i=0; i<nbr_diff_chunks; ++i) mpi_packet[i+3+nbr_diff_chunks] = chunks_nbr[i];
                for (int i=0; i<3; ++i) mpi_packet[i+3+nbr_diff_chunks*2] = chunks_per_dims[i];
                MPI_Bcast ((void*)mpi_packet, 6+MAX_DIFF_CHUNKS*2, MPI_INTEGER, 0, MPI_COMM_WORLD);
                
                MPI_Datatype old_types[2] = {MPI_INTEGER, mpi_datum_type};
                MPI_Aint steps[2] = {0, 6*sizeof(int)};
                int blocks_sizes[2]; blocks_sizes[0] = 6;

                mpi_types = new MPI_Datatype[3];
                for (int i=0; i<nbr_diff_chunks; ++i) {
                    blocks_sizes[1] = chunks_len[i];
                    MPI_Type_struct (2, blocks_sizes, steps, old_types, mpi_types+i);
                }
            }
            RES_T mpiRegisterSlave () {
                int *mpi_packet = new int[6+MAX_DIFF_CHUNKS*2];
                MPI_Bcast ((void*)mpi_packet, 6+MAX_DIFF_CHUNKS*2, MPI_INTEGER, 0, MPI_COMM_WORLD);
                nbr_diff_chunks = mpi_packet[0];
                nbr_chunks = mpi_packet[1];
                mpi_datum_type = mpi_packet[2];
                chunks_len = new int[nbr_diff_chunks];
                for (int i=0; i<nbr_diff_chunks; ++i) chunks_len[i] = mpi_packet[i+3];
                chunks_nbr = new int[nbr_diff_chunks];
                for (int i=0; i<nbr_diff_chunks; ++i) chunks_nbr[i] = mpi_packet[i+3+nbr_diff_chunks];
                for (int i=0; i<3; ++i) chunks_per_dims[i] = mpi_packet[i+3+nbr_diff_chunks*2];
                mpi_types = new MPI_Datatype[nbr_diff_chunks];

                MPI_Datatype old_types[2] = {MPI_INTEGER, mpi_datum_type};
                MPI_Aint steps[2] = {0, 6*sizeof(int)};
                int blocks_sizes[2]; blocks_sizes[0] = 6;

                mpi_types = new MPI_Datatype[3];
                for (int i=0; i<nbr_diff_chunks; ++i) {
                    blocks_sizes[1] = chunks_len[i];
                    MPI_Type_struct (2, blocks_sizes, steps, old_types, mpi_types+i);
                }
            }
            RES_T mpiFree () {
                for (int i=0; i<nbr_diff_chunks; ++i) {
                    MPI_Type_free (mpi_types+i) ;
                }
            }
            const int get_nbr_diff_chunks () const {
                return nbr_diff_chunks;
            }
            const int get_chunks_len (unsigned int i) const {
                ASSERT (i < nbr_diff_chunks);
                return chunks_len[i];
            }    
            const int get_chunks_nbr (unsigned int i) const {
                ASSERT (i < nbr_diff_chunks);
                return chunks_nbr[i];
            }
    };
}

#endif
