#include <DRecver.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "usage : mpiexec <bin> <path> <ip_address> <ip_port>" << endl;
        return -1;
    } 

    // Communication canal ...
    MPI_Comm inter_PtoR, intra_PtoR;
    // Rank ...
    int rank_in_PtoR;
    // World count ...
    int nbrP;
    // Service name ...
    char service[] = "smil_mpi_PtoR";
    // Port names ...
    char port_PtoR[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    stringstream ss;
    ss << "tag#1$description#" << argv[2] << "$port#" << argv[3] << "$ifname#" << argv[2] << "$" << endl;
    ss >> port_PtoR;

    cout << "Connecting to : " << port_PtoR << "...";

    int err, err_str_len; char err_str[256] ={};
    if (/*(err = MPI_Lookup_name (service, info, port_PtoR)) ||*/ 
        (err = MPI_Comm_connect (port_PtoR, info, 0, MPI_COMM_WORLD, &inter_PtoR)) ) { 
        MPI_Error_string (err, err_str, &err_str_len);
        cerr << "Connection to \'" << port_PtoR << "\' has failed ... aborting (" << err_str << ")." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }
    cout << "OK" << endl;

    MPI_Comm_remote_size (inter_PtoR, &nbrP);
    // Flag false is needed, so that the sender is always at rank = 0.
    MPI_Intercomm_merge (inter_PtoR, false, &intra_PtoR);
    MPI_Comm_rank (intra_PtoR, &rank_in_PtoR);
    
    GlobalHeader gh;

    broadcastMPITypeRegistration (gh, intra_PtoR, 1, "UINT8");
    Image<UINT8> im = Image<UINT8>(gh.size[0],gh.size[1],gh.size[2]);
    RecvStream<UINT8> rs (im);
    // Could create here multiple RecvBuffer and attach them to different process P.
    RecvBuffer<UINT8> rb (gh);

    // Main loop, where reception and writing to the array is done with the use of OpenMP.
    rb.loop (intra_PtoR, rank_in_PtoR, rs);

    freeMPIType (gh) ;
    write (im, argv[1]) ;

    cout << "Receiver terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
