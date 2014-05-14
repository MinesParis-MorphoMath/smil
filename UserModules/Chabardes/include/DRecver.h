#ifndef _DRECV_H_
#define _DRECV_H_

#include "DChabardes.h"
#include "DRecvStream.h"
#include "DRecvBuffer.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const int comm, const int root) {
        ASSERT (!gh.is_initialized) ;
        int packet[3];

        MPI_Recv ((void*)packet, 3, MPI_INTEGER, root, PTOR_MPITYPEREGISTRATION_TAG, comm, MPI_STATUS_IGNORE);

        gh.nbr_chunks = packet[0];
        gh.chunk_len = packet[1];
        gh.mpi_datum_type = packet[2];

        MPI_Datatype old_types[2] = {MPI_INTEGER, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
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
