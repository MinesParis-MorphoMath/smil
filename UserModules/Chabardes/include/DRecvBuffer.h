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
                if (rawData != NULL)
                    ::operator delete (rawData);
                if (ca != NULL)
                    delete[] ca;
            }
            RES_T initialize (const GlobalHeader& gh) {
                int nthreads = Core::getInstance()->getNumberOfThreads ();
                size = nthreads;

                memory_step = 12*sizeof(int) + gh.chunk_len*sizeof(T);
                if (rawData == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                ca = new Chunk<T> [nthreads];
                if (ca == NULL)
                    return RES_ERR_BAD_ALLOCATION;
                for (int i=0; i<size; ++i)
                    ca[i].setMemorySpace ((unsigned char*)rawData+i*memory_step, memory_step);
                return RES_OK;
            }
            RES_T loop(const MPI_Comm &comm, const int rank, RecvStream<T> &rs) {
                #ifdef USE_OPENMP
                int tid;
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
        private:
            int memory_step;
            int size;
            Chunk<T>* ca;
            void *rawData;
    };

}

#endif
