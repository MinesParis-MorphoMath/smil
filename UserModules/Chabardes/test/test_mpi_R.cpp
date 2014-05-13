#include <DReceiver.h>

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
    char service[] = "smil_mpi_PtoR";
    // Port names ...
    char port_PtoR[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    stringstream ss;
    ss << "tag#1$description#192.168.220.108$port#" << argv[1] << "$ifname#" << "192.168.220.108" << "$" << endl;
    ss >> port_PtoR;

    cout << "Connecting to : " << port_PtoR << "..." << endl;

    if (MPI_Lookup_name (service, info, port_PtoR) || MPI_Comm_connect (port_PtoR, info, 0, MPI_COMM_WORLD, &inter_PtoR) ) {
        cerr << "Connection to \"" << port_PtoR << "\" has failed ... aborting." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    Image<UINT8> im;
    GlobalHeader gh (im);
    RecvStream<UINT8> rs (gh);
    RecvBuffer<UINT8> rb (gh);

    broadcastMPITypeRegistration (inter_PtoR);

    rb.next (); 
    while (!isEndOfTransmission (rb)) {
        rb.write (rs);
        rb.next ();
    }

    cout << "Receiver terminates..." << endl;

    MPI_Finalize ();

    return 0;
}
