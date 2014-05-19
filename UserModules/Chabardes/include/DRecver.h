#ifndef _DRECV_H_
#define _DRECV_H_

#include "DChabardes.h"
#include "DRecvStream.h"
#include "DRecvBuffer.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const int comm, const int root) {
        ASSERT (!gh.is_initialized) ;
        unsigned int packet[6];

        MPI_Recv ((void*)packet, 6, MPI_UNSIGNED, root, PTOR_MPITYPEREGISTRATION_TAG, comm, MPI_STATUS_IGNORE);

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
    
    template <class T>
    bool isEndOfTransmission (const RecvBuffer<T> &rb) {

    }

    RES_T freeMPIType (GlobalHeader &gh) {
        ASSERT (gh.is_initialized); 
        MPI_Type_free (&(gh.mpi_type)) ;
    }
}

#endif 
