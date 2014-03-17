#include "DCore.h"

using namespace smil;

int main () {

    CpuID id; 

    cout << "Vendor: " << id.getVendor () << endl;
    cout << "Cores: " << id.getCores () << endl;
    cout << "Logical: " << id.getLogical () << endl;
    cout << "HyperThreading: " << id.isHyperThreated () << endl;

    SIMD_Instructions si = id.getSimdInstructions () ;

    cout << "MMX: " << si.MMX << endl;
    cout << "SSE: " << si.SSE << endl;
    cout << "SSE2: " << si.SSE2 << endl;
    cout << "SSE2: " << si.SSE3 << endl;
    cout << "SSSE3: " << si.SSSE3 << endl;
    cout << "SSE41: " << si.SSE41 << endl;
    cout << "SSE42: " << si.SSE42 << endl;
    cout << "AES: " << si.AES << endl;
    cout << "AVX: " << si.AVX << endl;

    Cache_Descriptors L1 = *id.getCacheAtLevel(0), L2 = *id.getCacheAtLevel(1), L3 = *id.getCacheAtLevel(2);

    cout << "L1.size: " << L1.size << endl;
    cout << "L1.associativity: " << L1.associativity << endl;
    cout << "L1.lines_per_tag: " << L1.lines_per_tag << endl;
    cout << "L1.line_size: " << L1.line_size << endl;
    cout << "L2.size: " << L2.size << endl;
    cout << "L2.associativity: " << L2.associativity << endl;
    cout << "L2.lines_per_tag: " << L2.lines_per_tag << endl;
    cout << "L2.line_size: " << L2.line_size << endl;
    cout << "L3.size: " << L3.size << endl;
    cout << "L3.associativity: " << L3.associativity << endl;
    cout << "L3.lines_per_tag: " << L3.lines_per_tag << endl;
    cout << "L3.line_size: " << L3.line_size << endl;

    return 0;
}
