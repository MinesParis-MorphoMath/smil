#ifndef _DGLOBALHEADER_H_
#define _DGLOBALHEADER_H_

namespace smil {

    class GlobalHeader {
        private:
            int nbr_chunks;
            int chunk_len;
            int chunks_per_dim[3];
            // mpi_specifics.
            int mpi_datum_type;
            int mpi_type;
            // Not transmitted.
            bool is_initialized;
        public:
            GlobalHeader () : is_initialized (false) {}
    };

}

#endif
