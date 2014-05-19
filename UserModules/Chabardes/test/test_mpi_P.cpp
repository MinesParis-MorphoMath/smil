#include <DProcessing.h>
#include <DSharedImage.hpp>

using namespace smil;

void processChunk (Chunk<UINT8> &c, const MPI_Comm &comm, const int rank, const GlobalHeader& gh) {
    SharedImage<UINT8> fakeIm (c.getData(), c.getSize(0), c.getSize(1), c.getSize(2));
    Image<UINT8> tmp = Image<UINT8> (fakeIm);

    erode (fakeIm, tmp, Cross3DSE());
    dilate (fakeIm, fakeIm, Cross3DSE());
    fakeIm -= tmp;
}

int main (int argc, char* argv[]) {

    if (argc != 1) {
        cerr << "usage : mpiexec <bin>" << endl;
        return -1;
    }

    // Communication canal ...
    MPI_Comm intra_P=MPI_COMM_WORLD, inter_StoP, inter_PtoR, intra_StoP, intra_PtoR;
    // Ranks ...
    int rank_inP, rank_in_StoP, rank_in_PtoR;
    // World count ...
    int nbrP = 16;
    // Service name ...
    char service_StoP[] = "smil_mpi_StoP", service_PtoR[] = "smil_mpi_PtoR";
    // Port names ...
    char port_StoP[MPI_MAX_PORT_NAME] = {0}, port_PtoR[MPI_MAX_PORT_NAME] = {0};
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nbrP);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank_inP);  

    if (rank_inP == 0) {
            
        // Not implemented on either mpich2 nor openmpi... using "mpiexec -host <ip_address> <bin>"
        /*
        MPI_Info_create (&info) ;
        MPI_Info_set (info, "ip_port", argv[1]);
        MPI_Info_set (info, "ip_address", argv[2]);
        */

        MPI_Open_port (info, port_StoP);
        cout << "Opened a sender port at : " << port_StoP << "." << endl;
//        MPI_Publish_name (service_StoP, info, port_StoP);
        
        MPI_Open_port (info, port_PtoR);
        cout << "Opened a receiver port at : " << port_PtoR << "." << endl;
//        MPI_Publish_name (service_PtoR, info, port_PtoR);
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (port_StoP, MPI_MAX_PORT_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast (port_PtoR, MPI_MAX_PORT_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank_inP == 0)
        cout << "Waiting for a Sender to connect...";
    MPI_Comm_accept (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP);
    if (rank_inP == 0)
        cout << "OK" << endl << "Waiting for a Receiver to connect...";
    MPI_Comm_accept (port_PtoR, info, 0, MPI_COMM_WORLD, &inter_PtoR);
    if (rank_inP == 0)
        cout << "OK" << endl;

    MPI_Barrier (MPI_COMM_WORLD) ;

    if (rank_inP == 0) {
        MPI_Close_port (port_StoP);
        MPI_Close_port (port_PtoR);
//        MPI_Unpublish_name (service_StoP, info, port_StoP);
//        MPI_Unpublish_name (service_PtoR, info, port_StoP);
    }

    MPI_Intercomm_merge (inter_StoP, true, &intra_StoP) ;
    MPI_Intercomm_merge (inter_PtoR, true, &intra_PtoR) ;
    MPI_Comm_rank (intra_StoP, &rank_in_StoP) ;
    MPI_Comm_rank (intra_PtoR, &rank_in_PtoR) ;

    struct timeval tvalB, tvalA;
    if (rank_inP == 0) {
        gettimeofday (&tvalB, NULL);
    }

    GlobalHeader gh;
    Chunk<UINT8> c;

    broadcastMPITypeRegistration (gh, intra_StoP, 0, intra_PtoR, 1, rank_in_PtoR, 0);

    int memory_size = 12*sizeof(int) + gh.chunk_len*sizeof(UINT8);
    void *rawData = ::operator new (memory_size);
    if (rawData == NULL) {
        cerr << "Unable to allocate memory..." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }
    c.setMemorySpace (rawData, memory_size, gh.mpi_type);
    MPI_Status status;

    do {
        c.recv (0, MPI_ANY_TAG, intra_StoP, &status); 
        if (status.MPI_TAG == CHUNK_TAG) {
            processChunk (c, intra_P, rank_inP, gh);
            c.send (0, CHUNK_TAG, intra_PtoR);
        }
    } while (status.MPI_TAG != EOT_TAG);
   
    MPI_Barrier (MPI_COMM_WORLD);
    // Propagation of End Of Transmittion
    if (rank_inP == 0) {
        c.send (0, EOT_TAG, intra_PtoR); 
    }
    ::operator delete (rawData);
    freeMPIType (gh) ;

    MPI_Barrier (MPI_COMM_WORLD);

    if (rank_inP == 0) {
        gettimeofday (&tvalA, NULL) ;
        cout << "time spent : " << (float)(tvalA.tv_sec - tvalB.tv_sec) << "s." << endl;
    }

    MPI_Comm_disconnect (&inter_StoP);
    MPI_Comm_disconnect (&inter_PtoR);
    MPI_Finalize ();

    if (rank_inP == 0)
        cout << "Processing terminates..." << endl;

    return 0;
} 
