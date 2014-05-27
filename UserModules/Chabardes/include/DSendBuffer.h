#ifndef _DSENDBUFFER_H_
#define _DSENDBUFFER_H_

namespace smil{
    template <class T>
    class SendBuffer {
        private:
            unsigned long memory_step;
            Chunk<T> c;
            void *rawData;
        public:
            SendBuffer (const GlobalHeader &gh) {
                initialize (gh);
            }
            ~SendBuffer () {
                if (rawData != NULL)
                    ::operator delete (rawData);
            }
            RES_T initialize (const GlobalHeader &gh) {
                memory_step = 12 * sizeof (unsigned int) + gh.chunk_len * sizeof(T);
                rawData = ::operator new (memory_step) ;
                if (rawData == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                c.setMemorySpace ((unsigned char*)rawData, memory_step, gh.mpi_type) ;
                return RES_OK;
            }
            RES_T loop (const MPI_Comm &comm, const int rank, const GlobalHeader &gh, SendStream<T> &ss) {
                int nbr_dest;
                MPI_Comm_size (comm, &nbr_dest) ;
                nbr_dest--;

                for (int i=0; i<gh.nbr_chunks; ++i) {
                    ss.read_at (i, c);
                    c.send ((i%nbr_dest)+1, CHUNK_TAG, comm) ;
                }

                // sending EOT to every processing units.
                for (int i=1; i<nbr_dest+1; ++i) {
                    c.send (i, EOT_TAG, comm); 
                }
            }
    };
}

#endif
