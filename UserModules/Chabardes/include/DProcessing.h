#ifndef _DPROCESSING_H_
#define _DPROCESSING_H_

#ifdef USE_OPEN_MP
#undef USE_OPEN_MP
#endif

#include "DChabardes.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const int intra_StoP, const int root_StoP, const int inter_PtoR, const int root_PtoR, const int rank_in_PtoR, const int dest_PtoR) {
        ASSERT (!gh.is_initialized) ;
        unsigned int packet[6];

        MPI_Bcast ((void*)packet, 6, MPI_UNSIGNED, root_StoP, intra_StoP);
        if (rank_in_PtoR == root_PtoR)
            MPI_Send ((void*)packet, 6, MPI_UNSIGNED, dest_PtoR, PTOR_MPITYPEREGISTRATION_TAG, inter_PtoR);

        gh.size[0] = packet[0];
        gh.size[1] = packet[1];
        gh.size[2] = packet[2];
        gh.nbr_chunks = packet[3];
        gh.chunk_len = packet[4];
        gh.mpi_datum_type = packet[5];


        MPI_Datatype old_types[2] = {MPI_UNSIGNED, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(unsigned int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
        MPI_Type_commit (&(gh.mpi_type));
        gh.is_initialized = true;
    }

    RES_T broadcastEndOfTransmission (const MPI_Comm &comm, const GlobalHeader &gh) {
        MPI_Send (NULL, 1, gh.mpi_type, 0, EOT_TAG, comm);
    }

    RES_T freeMPIType (GlobalHeader &gh) {
        ASSERT (gh.is_initialized);
        MPI_Type_free (&(gh.mpi_type)) ;
    }
}

#endif
