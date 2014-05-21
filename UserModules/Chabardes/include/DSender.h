#ifndef _DSENDER_H_
#define _DSENDER_H_

#include "DChabardes.h"
#include "DSendStream.h"
#include "DSendBuffer.h"

namespace smil {
    template <class T>
    RES_T initialize (const unsigned int nbr_procs, const int intersect_width, Image<T> &im, GlobalHeader &gh, SendArrayStream_chunk<T> &ss) {
        size_t s[3];
        im.getSize (s);
        gh.mpi_datum_type = smilToMPIType (im.getTypeAsString());

        for (int i=0; i<3; ++i ) ss.size[i] = s[i];
        for (int i=0; i<3; ++i ) gh.size[i] = s[i];
        ss.data = im.getPixels ();
        ss.intersect_width = intersect_width;
        ss.chunks_per_dim[0] = 1;
        ss.chunks_per_dim[1] = 1;
        ss.chunks_per_dim[2] = 1;

        while (ss.chunks_per_dim[0]*ss.chunks_per_dim[1]*ss.chunks_per_dim[2] < nbr_procs) {
            if (ss.size[0]/ss.chunks_per_dim[0] >= ss.size[1]/ss.chunks_per_dim[1] &&
                ss.size[0]/ss.chunks_per_dim[0] >= ss.size[2]/ss.chunks_per_dim[2])
                ss.chunks_per_dim[0]++;
            else if (ss.size[1]/ss.chunks_per_dim[1] >= ss.size[2]/ss.chunks_per_dim[2]) 
                ss.chunks_per_dim[1]++;
            else 
                ss.chunks_per_dim[2]++;
        }
        if (ss.size[0]%ss.chunks_per_dim[0] != 0) {ss.chunks_per_dim[0]++;}
        if (ss.size[1]%ss.chunks_per_dim[1] != 0) {ss.chunks_per_dim[1]++;}
        if (ss.size[2]%ss.chunks_per_dim[2] != 0) {ss.chunks_per_dim[2]++;}

        gh.nbr_chunks = ss.chunks_per_dim[0]*ss.chunks_per_dim[1]*ss.chunks_per_dim[2];
        gh.chunk_len = (ss.size[0]/ss.chunks_per_dim[0]+intersect_width*2)*
                       (ss.size[1]/ss.chunks_per_dim[1]+intersect_width*2)*
                       (ss.size[2]/ss.chunks_per_dim[2]+intersect_width*2); 
        ;

        ss.nbr_chunks = gh.nbr_chunks;
        ss.chunk_len = gh.chunk_len; 
        gh.is_initialized = true;
    }

    template <class T>
    RES_T initialize (const unsigned int nbr_procs, Image<T> &im, GlobalHeader &gh, SendArrayStream_slice<T> &ss) {

    }

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const int &comm, const int &rank) {
        ASSERT (gh.is_initialized) ;
        unsigned int packet[6];
        packet[0] = gh.size[0];
        packet[1] = gh.size[1];
        packet[2] = gh.size[2];
        packet[3] = gh.nbr_chunks;
        packet[4] = gh.chunk_len;
        packet[5] = gh.mpi_datum_type;

        MPI_Bcast ((void*)packet, 6, MPI_UNSIGNED, rank, comm);

        MPI_Datatype old_types[2] = {MPI_UNSIGNED, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(unsigned int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
        MPI_Type_commit (&(gh.mpi_type));
    }

    RES_T freeMPIType (GlobalHeader &gh) {
        ASSERT (gh.is_initialized); 
        MPI_Type_free (&(gh.mpi_type)) ;
    }
}
#endif
