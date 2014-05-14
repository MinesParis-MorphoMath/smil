#ifndef _DSENDBUFFER_H_
#define _DSENDBUFFER_H_

namespace smil{
    template <class T>
    class SendBuffer {
        private:
            int memory_step;
            int size;
            Chunk<T> *ca;
            void *rawData;
        public:
            SendBuffer (const int nbr_procs, const GlobalHeader &gh) {
                initialize (nbr_procs, gh);
            }
            ~SendBuffer () {
                if (rawData != NULL)
                    ::operator delete (rawData);
                if (ca != NULL)
                    delete[] ca;
            }
            RES_T initialize (const int nbr_procs, const GlobalHeader &gh) {
                size = nbr_procs;
                memory_step = 12 * sizeof (int) + gh.chunk_len * sizeof(T);
                rawData = ::operator new (size*memory_step) ;
                if (rawData == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                ca = new Chunk<T>[size];
                if (ca == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                for (int i=0; i<size; ++i)
                    ca[i].setMemorySpace ((unsigned char*)rawData+i*memory_step, memory_step) ;
                return RES_OK;
            }
            RES_T loop (const MPI_Comm &comm, const int rank, SendStream<T> &ss) {
                #ifdef USE_OPENMP
                int tid;
                int nthreads = Core::getInstance()->getNumberOfThreads ();
                #endif

                #ifdef USE_OPENMP
                #pragma omp parallel private(tid)
                #endif
                {
                    #ifdef USE_OPENMP
                    tid = omp_get_thread_num();
                    #endif
                }
            }
    };
}

#endif
