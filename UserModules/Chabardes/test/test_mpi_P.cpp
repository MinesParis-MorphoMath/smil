#include <DProcessing.h>

using namespace smil;

int main (int argc, char* argv[]) {
    // Communication canal ...
    MPI_Comm intraP, inter_StoP, inter_PtoR;
    // Ranks ...
    int rank_inP;
    // World count ...
    int nbrS, nbrP, nbrR, nbrTot;
    // Service name ...
    char service[] = "smil_mpi_stop";
    // Port names ...
    char port_StoP[MPI_MAX_PORT_NAME], port_PtoR[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nbrP);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank_inP); 

    if (rank_inP == 0) {
        MPI_Open_port (info, port_StoP);
        MPI_Open_port (info, port_PtoR);

        cout << "Processing nodes... Opened a senders port at : " << port_StoP << "." << endl;
        cout << "Processing nodes... Opened a receivers port at : " << port_PtoR << "." << endl;

        MPI_Publish_name (service, info, port_StoP);
        MPI_Publish_name (service, info, port_PtoR);

        MPI_Comm_accept (port_StoP, info, rank_inP, MPI_COMM_WORLD, &inter_StoP);
        MPI_Comm_accept (port_PtoR, info, rank_inP, MPI_COMM_WORLD, &inter_PtoR);
    }

    int j=0;
    for (int i=0; i<10000000; ++i) {
        j++;
    }

    if (rank_inP == 0) {
        MPI_Comm_disconnect (&inter_StoP);
        MPI_Comm_disconnect (&inter_PtoR);

        MPI_Unpublish_name (service, info, port_StoP);
        MPI_Unpublish_name (service, info, port_PtoR);
        MPI_Close_port (port_StoP);
        MPI_Close_port (port_PtoR);
    }

    MPI_Finalize ();

    return 0;
} 
