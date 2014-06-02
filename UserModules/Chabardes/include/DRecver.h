#ifndef _DRECV_H_
#define _DRECV_H_

#include "DChabardes.h"
#include "DRecvStream.h"
#include "DRecvBuffer.h"

namespace smil {

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const MPI_Comm comm, const int root) {
        ASSERT (!gh.is_initialized) ;
        gh.declareGHType ();
        void* packet = ::operator new (4*sizeof(unsigned int)+sizeof(unsigned long)+16*sizeof(char));

        MPI_Recv (packet, 1, gh.this_type, root, MPITYPEREGISTRATION_TAG, comm, MPI_STATUS_IGNORE);

        gh.size[0] = *((unsigned int*)packet);
        gh.size[1] = *((unsigned int*)packet+1);
        gh.size[2] = *((unsigned int*)packet+2);
        gh.nbr_chunks = *((unsigned int*)packet+3);
        gh.datum_size = *((unsigned int*)packet+4);
        gh.chunk_len = *((unsigned long*)((unsigned int*)packet+5));
        memcpy(gh.datum_type, (void*)((unsigned long*)((unsigned int*)packet+5)+1), 16*sizeof(char));
        gh.mpi_datum_type = smilToMPIType(gh.datum_type);

        ::operator delete(packet); 

        MPI_Datatype old_types[2] = {MPI_UNSIGNED, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(unsigned int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
        MPI_Type_commit (&(gh.mpi_type));
        gh.is_initialized = true;
    }
    template<class T>
    class recver {
        public:
            recver (bool verbose=false) : info(MPI_INFO_NULL), is_connected(false), is_verbose(false) {
                memset (port_PtoR, 0, MPI_MAX_PORT_NAME) ;
            }
            RES_T connect (string address) {
                int is_initialized = false;
                MPI_Initialized (&is_initialized);
                if (!is_initialized){
                    cout << "MPI is not initialized." << endl;
                    return RES_ERR_UNKNOWN;
                }
                strcpy (port_PtoR, address.c_str());
                if (is_verbose)
                    cout << "Connecting to : " << port_PtoR << "...";
                
                int err, err_str_len; char err_str[256] ={};
                if (err = MPI_Comm_connect (port_PtoR, info, 0, MPI_COMM_WORLD, &inter_PtoR))  { 
                    MPI_Error_string (err, err_str, &err_str_len);
                    cerr << "Connection to \'" << port_PtoR << "\' has failed ... aborting (" << err_str << ")." << endl;
                    MPI_Abort (MPI_COMM_WORLD, -1);
                }
                if (is_verbose)
                    cout << "OK" << endl;
                MPI_Comm_remote_size (inter_PtoR, &nbrP);
                MPI_Intercomm_merge (inter_PtoR, false, &intra_PtoR);
                MPI_Comm_rank (intra_PtoR, &rank_in_PtoR);
                is_connected = true;
                return RES_OK;
            }
            RES_T run (Image<T> &im) {
                ASSERT (is_connected);
                broadcastMPITypeRegistration (gh, intra_PtoR, 1);
                im = Image<T> (gh.size[0], gh.size[1], gh.size[2]);
                
                RecvStream<T> rs (im);
                // Could create here multiple RecvBuffer and attach them to different process P.
                RecvBuffer<T> rb (gh);

                // Main loop, where reception and writing to the array is done with the use of OpenMP.
                rb.loop (intra_PtoR, rank_in_PtoR, rs);
                return RES_OK;
            }
            RES_T disconnect () {
                ASSERT (is_connected);
                MPI_Comm_disconnect (&inter_PtoR) ;
                gh.freeMPIType ();
                is_connected = false;
                return RES_OK;
            }
            void printInfo () {
                if (!is_connected) {
                    cout << "Process recver is not connected." << endl;
                    return;
                }
                cout << "\tPtoR port : " << port_PtoR << endl;        
                cout << "\tNumber of cores : " << nbrP << endl;
            }
            void verbose () {
                is_verbose = true;
            }
            void quiet () {
                is_verbose = false;    
            }
        private:
            // Communication canal .
            MPI_Comm inter_PtoR, intra_PtoR;
            // Rank .
            int rank_in_PtoR;
            // World count .
            int nbrP;
            // Port names .
            char port_PtoR[MPI_MAX_PORT_NAME];
            // MPI Implementation specific information on how to establish an address. 
            MPI_Info info;
            // Others ... 
            bool is_connected;
            bool is_verbose;
            GlobalHeader gh;
    };
}

#endif 
