#include <DSender.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "usage : mpiexec <bin> <ip_port> <ip_address>" << endl;
        return -1;
    } 

    // Communication canal ...
    MPI_Comm inter_StoP, intra_StoP;
    // Rank ...
    int rank_in_StoP;
    // World count ...
    int nbrP;
    // Service name ...
    char service[] = "smil_mpi_StoP";
    // Port names ...
    char port_StoP[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    stringstream strs;
    strs << "tag#0$description#" << argv[2] << "$port#" << argv[1] << "$ifname#" << argv[2] << "$" << endl;
    strs >> port_StoP;

    cout << "Connecting to : " << port_StoP << "...";

    if (/*MPI_Lookup_name (service, info, port_StoP) ||*/
         MPI_Comm_connect (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP) ) {
        cerr << "Connection to \"" << port_StoP << "\" has failed ... aborting." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }
    cout << "OK" << endl;

    MPI_Comm_remote_size (inter_StoP, &nbrP);
    // Flag false is needed, so that the sender is always at rank = 0.
    MPI_Intercomm_merge (inter_StoP, false, &intra_StoP);
    MPI_Comm_rank (intra_StoP, &rank_in_StoP);

    Image<UINT8> im = Image<UINT8> (300,300,300) ;
    GlobalHeader gh;
    // Choosing a chunk-style partionning.
    SendArrayStream_chunk<UINT8> ss;

    initialize (nbrP, 1, im, gh, ss) ;

    // Could create here multiple SendBuffer and attach them to different process P.
    SendBuffer<UINT8> sb(nbrP, gh); 

    broadcastMPITypeRegistration (gh, intra_StoP, rank_in_StoP);

    // Main loop, where reading and sending from the array is done with the use of OpenMP.
    sb.loop (intra_StoP, rank_in_StoP, ss);

    freeMPIType (gh) ;

    cout << "Sender terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
