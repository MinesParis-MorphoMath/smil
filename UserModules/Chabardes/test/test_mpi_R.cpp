#include <DRecver.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "usage : mpiexec <bin> <ip_port> <ip_address>" << endl;
        return -1;
    } 

    // Communication canal ...
    MPI_Comm inter_PtoR, intra_PtoR;
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
    ss << "tag#1$description#" << argv[2] << "$port#" << argv[1] << "$ifname#" << "192.168.220.108" << "$" << endl;
    ss >> port_PtoR;

    cout << "Connecting to : " << port_PtoR << "..." << endl;

    int err, err_str_len; char err_str[256] ={};
    if (/*(err = MPI_Lookup_name (service, info, port_PtoR)) ||*/ 
        (err = MPI_Comm_connect (port_PtoR, info, 0, MPI_COMM_WORLD, &inter_PtoR)) ) { 
        MPI_Error_string (err, err_str, &err_str_len);
        cerr << "Connection to \'" << port_PtoR << "\' has failed ... aborting (" << err_str << ")." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    MPI_Intercomm_merge (inter_PtoR, false, &intra_PtoR);

    MPI_Recv (&nbrP, 1, MPI_INT, 0, 1, inter_PtoR, MPI_STATUS_IGNORE) ;    
    cout << nbrP << endl;
/*
    Image<UINT8> im;
    GlobalHeader gh;

    broadcastMPITypeRegistration (gh, inter_PtoR);
    
    RecvStream<UINT8> rs (gh);
    RecvBuffer<UINT8> rb (gh);

    rb.next (); 
    while (!isEndOfTransmission (rb)) {
        rb.write (rs);
        rb.next ();
    }
*/
    cout << "Receiver terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
