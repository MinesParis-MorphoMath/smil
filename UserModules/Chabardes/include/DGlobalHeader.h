#ifndef _DGLOBALHEADER_H_
#define _DGLOBALHEADER_H_

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

    class GlobalHeader {
        public:
            int nbr_chunks;
            int chunk_len;
            // mpi_specifics.
            int mpi_datum_type;
            int mpi_type; // assigned when broadcastMPITypeRegistration is called.
            // Not transmitted.
            bool is_initialized;
            GlobalHeader () : is_initialized (false) {}
    };

}

#endif
