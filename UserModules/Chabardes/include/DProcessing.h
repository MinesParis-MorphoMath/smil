#ifndef _DPROCESSING_H_
#define _DPROCESSING_H_

#include "DChabardes.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const int intra_StoP, const int root_StoP, const int inter_PtoR, const int root_PtoR, const int rank_in_PtoR, const int dest_PtoR) {
        ASSERT (!gh.is_initialized) ;
        int packet[3];

        MPI_Bcast ((void*)packet, 3, MPI_INTEGER, root_StoP, intra_StoP);
        if (rank_in_PtoR == root_PtoR)
            MPI_Send ((void*)packet, 3, MPI_INTEGER, dest_PtoR, PTOR_MPITYPEREGISTRATION_TAG, inter_PtoR);

        gh.nbr_chunks = packet[0];
        gh.chunk_len = packet[1];
        gh.mpi_datum_type = packet[2];

        MPI_Datatype old_types[2] = {MPI_INTEGER, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
        gh.is_initialized = true;
    }

    RES_T broadcastEndOfTransmission (const MPI_Comm &comm) {

    }

    template <class T>
    RES_T send (const Chunk<T> &c, const GlobalHeader &gh, const MPI_Comm &comm) {

    }

    template <class T>
    RES_T recv (const Chunk<T> &c, const GlobalHeader &gh, const MPI_Comm &comm) {

    }

    RES_T freeMPIType (GlobalHeader &gh) {
        ASSERT (gh.is_initialized);
        MPI_Type_free (&(gh.mpi_type)) ;
    }
}

#endif
