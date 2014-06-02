#ifndef _DSENDER_H_
#define _DSENDER_H_

#include "DChabardes.h"
#include "DSendStream.h"
#include "DSendBuffer.h"

namespace smil {
    template <class T>
    RES_T initialize (const unsigned int nbr_procs, const unsigned int intersect_width, Image<T> &im, GlobalHeader &gh, SendArrayStream<T> &ss) {
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

    RES_T broadcastMPITypeRegistration (GlobalHeader &gh, const MPI_Comm &comm, const int &rank, const char* mpi_datum_type) {
        ASSERT (gh.is_initialized) ;
        gh.declareGHType ();
        void* packet = ::operator new (4*sizeof(unsigned int)+sizeof(unsigned long)+16*sizeof(char));

        *((unsigned int*)packet) = gh.size[0];
        *((unsigned int*)packet+1) = gh.size[1];
        *((unsigned int*)packet+2) = gh.size[2];
        *((unsigned int*)packet+3) = gh.nbr_chunks;
        *((unsigned int*)packet+4) = gh.datum_size;
        *((unsigned long*)((unsigned int*)packet+5)) = gh.chunk_len;
        memcpy((void*)((unsigned long*)((unsigned int*)packet+5)+1), gh.datum_type, 16*sizeof(char));
        gh.mpi_datum_type = smilToMPIType(gh.datum_type);

        MPI_Bcast (packet, 1, gh.mpi_datum_type, rank, comm);

        MPI_Datatype old_types[2] = {MPI_UNSIGNED, gh.mpi_datum_type};
        MPI_Aint steps[2] = {0, 12*sizeof(unsigned int)};
        int block_size[2]; block_size[0] = 12; block_size[1] = gh.chunk_len;

        MPI_Type_struct (2, block_size, steps, old_types, &(gh.mpi_type));
        MPI_Type_commit (&(gh.mpi_type));
    }

    template <class T>
    class sender {
        public :
            sender (bool verbose = false) : info (MPI_INFO_NULL), is_connected(false), is_verbose (false) {
                memset (port_StoP, 0, MPI_MAX_PORT_NAME) ;
            } 
            void connect (string ip_address, string ip_port) {
                int is_initialized = false;
                MPI_Initialized (&is_initialized);
                if (!is_initialized) {
                    cout << "MPI is not initialized." << endl;
                    return;
                }
                stringstream ss;
                ss << "tag#0$description#" << ip_address << "$port#" << ip_port << "$ifname#" << ip_address << "$" << endl;
                ss >> port_StoP;
                if (is_verbose)
                    cout << "Connecting to : " << port_StoP << "...";

                if (MPI_Comm_connect (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP) ) {
                    cerr << "Connection to \"" << port_StoP << "\" has failed ... aborting." << endl;
                    MPI_Abort (MPI_COMM_WORLD, -1);

                }
                if (is_verbose)
                    cout << "OK" << endl;

                MPI_Comm_remote_size (inter_StoP, &nbrP);
                MPI_Intercomm_merge (inter_StoP, false, &intra_StoP);
                MPI_Comm_rank (intra_StoP, &rank_in_StoP);
                is_connected = true;
            }
            RES_T run (Image<T> &im, const unsigned int min_nbr_blocs, const unsigned int intersect_width) {
                ASSERT (is_connected);
                SendArrayStream<T> ss;

                initialize (min_nbr_blocs, intersect_width, im, gh, ss) ;

                broadcastMPITypeRegistration (gh, intra_StoP, rank_in_StoP, gh.datum_type);

                SendBuffer<T> sb(gh);

                // Main loop, where reading and sending from the array is done with the use of OpenMP.
                sb.loop (intra_StoP, rank_in_StoP, gh, ss);
            }
            void disconnect () {
                MPI_Comm_disconnect (&inter_StoP);
                gh.freeMPIType ();
                is_connected = false;
            }
            void printInfo () {
                if (!is_connected) {
                    cout << "Process sender is not connected." << endl;
                    return;
                }
                cout << "\tStoP port : " << port_StoP << endl;        
                cout << "\tNumber of cores : " << nbrP << endl;
            }
            void verbose () {
                is_verbose = true;
            }
            void quiet () {
                is_verbose = false;    
            }
        private :
            // Communication canal .
            MPI_Comm inter_StoP, intra_StoP;
            // Rank .
            int rank_in_StoP;
            // World count .
            int nbrP;
            // Port names .
            char port_StoP[MPI_MAX_PORT_NAME];
            // MPI Implementation specific information on how to establish an address. 
            MPI_Info info;
            // Others ...
            bool is_connected;
            bool is_verbose;
            GlobalHeader gh;
    };
}

#endif
