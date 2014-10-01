#ifndef _DRECVBUFFER_H_
#define _DRECVBUFFER_H_

namespace smil {

    template <class T>
    class RecvBuffer {
        public:
            RecvBuffer (const GlobalHeader& gh) { 
                initialize (gh) ;
            }
            ~RecvBuffer () {
                if (rawdata != NULL)
                    ::operator delete (rawdata);
            }
            RES_T initialize (const GlobalHeader& gh) {
                memory_step = 12*sizeof(unsigned int) + gh.chunk_len*sizeof(T);
                rawdata = ::operator new (memory_step) ;
                if (rawdata == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                c.setMemorySpace ((unsigned char*)rawdata, memory_step, gh.mpi_type);
                return RES_OK;
            }
            RES_T loop(const MPI_Comm &comm, const int rank, RecvStream<T> &rs) {
                MPI_Status status;
                do {
                   c.recv (MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
                   if (status.MPI_TAG == CHUNK_TAG) {
                       rs.write (c) ;
                   }
                } while (status.MPI_TAG != EOT_TAG);
            }
        private:
            unsigned long memory_step;
            Chunk<T> c;
            void *rawdata;
    };

}

#endif
