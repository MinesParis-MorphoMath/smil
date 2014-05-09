#include <DRecver.h>

int main (int argc, char* argv[]) {
    // Communication canal ...
    MPI_Comm intraS, inter_StoP;
    // Ranks ...
    int rank_inS;
    // World count ...
    int nbrS, nbrP, nbrTot;
    // Service name ...
    char service[] = "smil_mpi";
    // Port names ...
    char port_StoP[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nbrS);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank_inS); 

    cout << "Please enter the port name of the processing communication node (max length : " << MPI_MAX_PORT_NAME << ") : " << endl;
    cin << port_StoP;

    if (MPI_Lookup_name (service, info, port_StoP)) {
        cerr << "Connection to  \"" << port_StoP << "\" has failed ... aborting." << endl;
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    MPI_Comm_connect (port_StoP, info, 0, MPI_COMM_WORLD, &inter_StoP);

    cout << "We can start processing the shit out of the processing communication node ." << endl;

    MPI_Finalize ();

    return 0;
}
