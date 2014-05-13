#include <DSender.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "usage : mpiexec <bin> <ip_port> <ip_address>" << endl;
        return -1;
    } 

    // Communication canal ...
    MPI_Comm inter_StoP, intra_StoP;
    // World count ...
    int nbrP;
    // Service name ...
    char service[] = "smil_mpi_StoP";
    // Port names ...
    char port_StoP[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    stringstream ss;
    ss << "tag#0$description#" << argv[2] << "$port#" << argv[1] << "$ifname#" << argv[2] << "$" << endl;
    ss >> port_StoP;

    cout << "Connecting to : " << port_StoP << "..." << endl;

    if (/*MPI_Lookup_name (service, info, port_StoP) ||*/
         MPI_Comm_connect (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP) ) {
        cerr << "Connection to \"" << port_StoP << "\" has failed ... aborting." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    MPI_Intercomm_merge (inter_StoP, false, &intra_StoP);

    MPI_Recv (&nbrP, 1, MPI_INT, 0, 0, inter_StoP, MPI_STATUS_IGNORE) ;

    cout << nbrP << endl;    
/*
    Image<UINT8> im;
    GlobalHeader gh;
    SendChunkStream<UINT8> ss;

    initializeChunkStyle (nbrP, 1, im, gh, ss) ;

    SendBuffer<UINT8> sb(gh); 

    broadcastMPITypeRegistration (gh, inter_StoP);

    do {
        sb.nextRound (ss);
        sb.scatter (inter_StoP);
    } while (!ss.eof());

    broadcastEndOfTransmission (inter_StoP);
*/
    cout << "Sender terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
