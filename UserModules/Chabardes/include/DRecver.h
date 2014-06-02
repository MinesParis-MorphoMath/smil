#ifndef _DRECV_H_
#define _DRECV_H_

#include "DChabardes.h"
#include "DRecvStream.h"
#include "DRecvBuffer.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const MPI_Comm comm, const int root, const char* mpi_datum_type) {
        ASSERT (!gh.is_initialized) ;
        declareGHType ();
        void* packet = ::operator new (4*sizeof(unsigned int)+sizeof(unsigned long)+16*sizeof(char));

        MPI_Recv (packet, 1, gh.this_type, root, PTOR_MPITYPEREGISTRATION_TAG, comm, MPI_STATUS_IGNORE);

        gh.size[0] = *((unsigned int)packet);
        gh.size[1] = *((unsigned int)packet+1);
        gh.size[2] = *((unsigned int)packet+2);
        gh.nbr_chunks = *((unsigned int)packet+3);
        gh.datum_size = *((unsigned int)packet+4);
        gh.chunk_len = *((unsigned long)((unsigned int)packet+5));
        memcpy(gh.datum_type, (void*)((unsigned long)((unsigned int)packet+5)+1), 16*sizeof(char));
        gh.mpi_datum_type = smilToMPIType(gh.datum_type);

        ::operator delete(packet); 

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
}

#endif 
