#include <DSender.h>

int main (int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "usage : mpiexec <bin> <ip_port>" << endl;
        return -1;
    } 

    // Communication canal ...
    MPI_Comm inter_StoP;
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
    ss << "tag#1$description#192.168.220.108$port#" << argv[1] << "$ifname#" << "192.168.220.108" << "$" << endl;
    ss >> port_StoP;

    cout << "Connecting to : " << port_StoP << "..." << endl;

    if (MPI_Lookup_name (service, info, port_StoP) || MPI_Comm_connect (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP) ) {
        cerr << "Connection to \"" << port_StoP << "\" has failed ... aborting." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    Image<UINT8> im;
    GlobalHeader gh(im);
    SendStream<UINT8> ss(gh);
    SendBuffer<UINT8> sb(gh); 

    broadcastMPITypeRegistration (inter_StoP);

    do {
        sb.nextRound (ss);
        sb.scatter (inter_StoP);
    } while (!ss.eof());

    broadCastEndOfTransmission (inter_StoP);

    cout << "Sender terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
