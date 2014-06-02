#include <DProcessing.h>
#include <DSharedImage.hpp>

using namespace smil;

int main (int argc, char* argv[]) {

    if (argc != 1) {
        cerr << "usage : mpiexec <bin>" << endl;
        return -1;
    }

    MPI_Init (&argc, &argv);

    cpu<UINT8> pc(true);
    pc.open_ports ();
    pc.accept_connection ();
    pc.run();
    pc.disconnect();

    MPI_Finalize ();

    return 0;
} 
