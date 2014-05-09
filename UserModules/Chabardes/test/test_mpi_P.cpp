#include <DChabardes.net>

using namespace smil;

int main (int argc, char* argv[]) {
    // Communication canal ...
    int intraP, inter_StoP, inter_PtoR;
    // Ranks ...
    int rank_inP;
    // World count ...
    int nbrS, nbrP, nbrR;
    // Port names
    char port_StoP[MPI_MAX_PORT_NAME], port_PtoR[MPI_MAX_PORT_NAME];
    // MPI Implementation specific information on how to establish an address. 
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nbrP);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank_inP); 

//    if (rank == 0) {
        MPI_Open_port (info, port_StoP);
        MPI_Open_port (info, port_PtoR);
        MPI_Publish_name ("smil_mpi_stop", info, &port_StoP);
        MPI_Publish_name ("smil_mpi_ptor", info, &port_PtoR);

        cout << "Processing nodes... Open a senders port with name : " << port_StoP << "." << endl;
        cout << "Processing nodes... Open a receivers port with name : " << port_PtoR << "." << endl;

        MPI_Comm_accept (&port_StoP, info, rank, MPI_COMM_WORLD, &inter_StoP);
        MPI_Comm_accept (&port_PtoR, info, rank, MPI_COMM_WORLD, &inter_PtoR);

        MPI_Comm_disconnect (inter_StoP);
        MPI_Comm_disconnect (inter_PtoR);

        MPI_Unpublish_name ("smil_mpi_stop", info, port_StoP);
        MPI_Unpublish_name ("smil_mpi_ptor", info, port_PtoR);
        MPI_Close_port (port_StoP);
        MPI_Close_port (port_PtoR);
//    }

    MPI_Finalyze ();

    return 0;
} 
