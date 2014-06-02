#ifndef _DPROCESSING_H_
#define _DPROCESSING_H_

#ifdef USE_OPEN_MP
#undef USE_OPEN_MP
#endif

#include "DChabardes.h"

namespace smil {
    template <class T>
    void processChunk (Chunk<T> &c, const MPI_Comm &comm, const int rank, const GlobalHeader& gh) {
        SharedImage<T> fakeIm (c.getData(), c.getSize(0), c.getSize(1), c.getSize(2));
        Image<T> tmp = Image<T> (fakeIm);

        erode (fakeIm, tmp, Cross3DSE());
        dilate (fakeIm, fakeIm, Cross3DSE());
        fakeIm -= tmp;
    }

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const MPI_Comm intra_StoP, const int root_StoP, const MPI_Comm inter_PtoR, const int root_PtoR, const int rank_in_PtoR, const int dest_PtoR) {
        ASSERT (!gh.is_initialized) ;
        gh.declareGHType ();
        void* packet = ::operator new (4*sizeof(unsigned int)+sizeof(unsigned long)+16*sizeof(char));

        MPI_Bcast (packet, 1, gh.this_type, root_StoP, intra_StoP);
        if (rank_in_PtoR == root_PtoR)
            MPI_Send (packet, 1, gh.this_type, dest_PtoR, MPITYPEREGISTRATION_TAG, inter_PtoR);

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

    RES_T broadcastEndOfTransmission (const MPI_Comm &comm, const GlobalHeader &gh) {
        MPI_Send (NULL, 1, gh.mpi_type, 0, EOT_TAG, comm);
    }

    template <class T>
    class cpu {
        public:
            cpu (bool verbose = false) : intra_P(MPI_COMM_WORLD), info(MPI_INFO_NULL), is_waiting_for_connection(false), is_ready(false), is_verbose (verbose){
                memset (port_StoP, 0, MPI_MAX_PORT_NAME);
                memset (port_PtoR, 0, MPI_MAX_PORT_NAME);
            }
            RES_T open_ports () {
                int is_initialized = false;
                MPI_Initialized (&is_initialized);
                if (!is_initialized) {
                    cout << "MPI is not initialized." << endl;
                    return RES_ERR_UNKNOWN;
                }
                MPI_Comm_size (intra_P, &nbrP);
                MPI_Comm_rank (intra_P, &rank_in_P);

                if (rank_in_P == 0) {
                    MPI_Open_port (info, port_StoP);
                    MPI_Open_port (info, port_PtoR);
                    if (is_verbose) {
                        cout << "Opened StoP port : " << port_StoP << endl;
                        cout << "Opened PtoR port : " << port_PtoR << endl;
                    }
                }
                MPI_Barrier (intra_P);
                MPI_Bcast (port_StoP, MPI_MAX_PORT_NAME, MPI_CHAR, 0, intra_P);
                MPI_Bcast (port_PtoR, MPI_MAX_PORT_NAME, MPI_CHAR, 0, intra_P);
                is_waiting_for_connection = true;
            }
            RES_T accept_connection () {
                ASSERT (is_waiting_for_connection);
                if (is_verbose && rank_in_P == 0)
                    cout << "Waiting for a sender to connect ...";
                MPI_Comm_accept (port_StoP, info, 0, intra_P, &inter_StoP);
                if (rank_in_P == 0) 
                    MPI_Close_port (port_StoP);
                if (is_verbose && rank_in_P == 0)
                    cout << "OK" << endl << "Waiting for a recver to connect ...";
                MPI_Comm_accept (port_PtoR, info, 0, intra_P, &inter_PtoR);
                if (is_verbose && rank_in_P == 0)
                    cout << "OK" << endl;
                if (rank_in_P == 0)
                    MPI_Close_port (port_PtoR);
                MPI_Barrier (intra_P);
                MPI_Intercomm_merge (inter_StoP, true, &intra_StoP);
                MPI_Intercomm_merge (inter_PtoR, true, &intra_PtoR);
                MPI_Comm_rank (intra_StoP, &rank_in_StoP);
                MPI_Comm_rank (intra_PtoR, &rank_in_PtoR);
                is_ready = true;
                return RES_OK;
            }
            RES_T disconnect () {
                ASSERT (is_ready);
                MPI_Barrier (intra_P);
                MPI_Comm_disconnect (&inter_StoP);
                MPI_Comm_disconnect (&inter_PtoR);
                if (is_verbose && rank_in_P == 0)
                    cout << "Disconnect." << endl;
                is_ready = false;
                is_waiting_for_connection = false;
                return RES_OK;
            }
            RES_T run () {
                ASSERT (is_ready);
                broadcastMPITypeRegistration (gh, intra_StoP, 0, intra_PtoR, 1, rank_in_PtoR, 0);
                Chunk<T> c;

                int memory_size = 12*sizeof(int) + gh.chunk_len*sizeof(gh.datum_size);
                void *rawData = ::operator new (memory_size);
                if (rawData == NULL){
                    cerr << "Unable to allocate memory..." << endl;
                    return RES_ERR_BAD_ALLOCATION;
                }
                c.setMemorySpace (rawData, memory_size, gh.mpi_type);

                MPI_Status status;

                do {
                    c.recv (0, MPI_ANY_TAG, intra_StoP, &status); 
                    if (status.MPI_TAG == CHUNK_TAG) {
                        processChunk (c, intra_P, rank_in_P, gh);
                        c.send (0, CHUNK_TAG, intra_PtoR);
                    }
                } while (status.MPI_TAG != EOT_TAG);
               
                MPI_Barrier (MPI_COMM_WORLD);
                // Propagation of End Of Transmittion
                if (rank_in_P == 0) {
                    c.send (0, EOT_TAG, intra_PtoR); 
                }
                ::operator delete (rawData);
                gh.freeMPIType();

                MPI_Barrier (MPI_COMM_WORLD);
                return RES_OK;
            }
            void printInfo () {
                if (rank_in_P != 0)
                    return;
                if (!is_waiting_for_connection ()){
                    cout << "Process cpu has not opened ports." << endl;
                    return;
                } else 
                    cout << "Process cpu has opened ports." << endl;
                cout << "\tStoP port : " << port_StoP << "." << endl;
                cout << "\tPtoR port : " << port_PtoR << "." << endl;
                cout << "\tNumber of cores : " << nbrP << "." << endl;
                if (!is_ready ()){
                    cout << "Process cpu is not ready to be run." << endl;
                }else
                    cout << "Process cpu is ready to be run." << endl;
            }
            void verbose () {
                is_verbose = true;
            }
            void quiet () {
                is_verbose = false;
            }
        private:
            // Communication canal.
            MPI_Comm intra_P, inter_StoP, inter_PtoR, intra_StoP, intra_PtoR;
            // Ranks.
            int rank_in_P, rank_in_StoP, rank_in_PtoR;
            // World count.
            int nbrP;
            // Port names.
            char port_StoP[MPI_MAX_PORT_NAME], port_PtoR[MPI_MAX_PORT_NAME];
            // MPI Implementation specific information on how to establish an address.
            MPI_Info info;
            // Others...
            bool is_waiting_for_connection;
            bool is_ready;
            bool is_verbose;
            GlobalHeader gh;
    };
}

#endif
