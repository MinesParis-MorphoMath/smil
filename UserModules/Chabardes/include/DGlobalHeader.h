#ifndef _DGLOBALHEADER_H_
#define _DGLOBALHEADER_H_

namespace smil {
    MPI_Datatype smilToMPIType (const char* datum_type) {
        if (strcmp(datum_type, "UINT8") == 0) {
                return MPI_UNSIGNED_CHAR;
        } else if (strcmp(datum_type, "UINT16") == 0) {
                return MPI_UNSIGNED_SHORT;
        } else if (strcmp(datum_type, "UINT32") == 0) {
                return MPI_UNSIGNED;
        } else if (strcmp(datum_type, "UINT64") == 0) {
                return MPI_UNSIGNED_LONG;
        } else if (strcmp(datum_type, "INT") == 0) {
                return MPI_INT;
        } else {
                return MPI_UNSIGNED;
        }
    }

    class GlobalHeader {
        public:
            GlobalHeader () : is_initialized (false) {}
            RES_T declareGHType () {
                MPI_Datatype old_types[3] = {MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_CHAR};
                MPI_Aint steps[3] = {0, 5*sizeof(unsigned int), 5*sizeof(unsigned int)+sizeof(unsigned long)};
                int block_size[3] = {5,1,16};
                MPI_Type_struct (3,block_size,steps,old_types,&this_type);
                MPI_Type_commit (&this_type);
                return RES_OK;
            }
            RES_T freeMPIType () {
                ASSERT (is_initialized); 
                MPI_Type_free (&mpi_type);
                MPI_Type_free (&this_type);
            }

            unsigned int size[3];
            unsigned int nbr_chunks;
            unsigned int datum_size;
            unsigned long chunk_len;
            char datum_type[16];
            // mpi_specifics.
            MPI_Datatype mpi_datum_type;
            MPI_Datatype mpi_type; // assigned when broadcastMPITypeRegistration is called.
            MPI_Datatype this_type;
            // Not transmitted.
            bool is_initialized;

    };

}

#endif
