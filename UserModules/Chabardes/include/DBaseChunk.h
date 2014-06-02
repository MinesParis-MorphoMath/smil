#ifndef _DBASECHUNK_H_
#define _DBASECHUNK_H_

namespace smil {

    class BaseChunk {
            public:
            BaseChunk (const char *_className) : is_initialized (false) { 
                strcpy (classname, _className);
            }
            ~BaseChunk () {
            }
            virtual RES_T setMemorySpace (void* ptr, const unsigned long _sent_size, MPI_Datatype dt) =0;
            const int getSize (const unsigned char &dimension) const {
                ASSERT (dimension < 3);
                return size[dimension];
            }
            unsigned int* getSize () {
                return size;
            }
            const unsigned int getOffset (const unsigned char &dimension) const {
                ASSERT (dimension < 3);
                return offset[dimension];
            }
            unsigned int* getOffset () {
                return offset;
            }
            const unsigned int getWrittenSize (const unsigned char &dimension) const {
                ASSERT (dimension < 3);
                return w_size[dimension];
            }
            unsigned int* getWrittenSize () {
                return w_size;
            }
            const unsigned int getWrittenOffset (const unsigned char &dimension) const {
                ASSERT (dimension < 3);
                return w_offset[dimension];
            }
            unsigned int* getWrittenOffset () {
                return w_offset;
            }
            unsigned int getRelativeWrittenOffset (const unsigned char &dimension) const {
                return w_offset[dimension] - offset[dimension];
            }
            const char * getTypeAsString () {
                return classname;
            }
            virtual void print (bool print_data = false) = 0;
            bool isInitialized () {
                return is_initialized;
            }
            RES_T send (const int dest, const int tag, const MPI_Comm &comm) {
                ASSERT (is_initialized);
                MPI_Send (rawData, 1, datatype, dest, tag, comm);
            } 
            RES_T recv (const int root, const int tag, const MPI_Comm &comm, MPI_Status* status) {
                ASSERT (is_initialized);
                MPI_Recv (rawData, 1, datatype, root, tag, comm, status);
            }

        protected:
            char classname [25] ;
            bool is_initialized;
            unsigned int* size;
            unsigned int* offset;
            unsigned int* w_size;
            unsigned int* w_offset;
            unsigned long sent_size;
            void* rawData;
            MPI_Datatype datatype;
    };
}

#endif
