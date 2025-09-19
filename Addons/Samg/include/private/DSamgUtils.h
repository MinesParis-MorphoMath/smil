#ifndef _D_UTILS_H_
#define _D_UTILS_H_

// #include <stdio.h>
// #include <sys/ioctl.h>
// #include <sys/select.h>
// #include <termios.h>
// // XXX JOE #include <stropts.h>
// #include <unistd.h>

namespace smil
{
  struct index_T {
    size_t x;
    size_t y;
    size_t z;
    size_t o;
    index_T()
    {
    }
    index_T(const index_T &i) : x(i.x), y(i.y), z(i.z), o(i.o)
    {
    }
  };

#define WS ImDtTypes<labelT>::max()
#define IndexToCoor(a)                                                         \
  a.z = a.o / nbrPixelsInSlice;                                                \
  a.y = (a.o % nbrPixelsInSlice) / S[0];                                       \
  a.x = a.o % S[0];
#define CoorToIndex(a) a.o = a.x + a.y * S[0] + a.z * nbrPixelsInSlice
#define ForEachPixel(a)                                                        \
  for (size_t i = 0; i < nbrPixels; ++i) {                                     \
    a.o = i;                                                                   \
    IndexToCoor(a);
#define ENDForEachPixel }
#define ForEachNeighborOf(a, b)                                                \
  for (pts = 0; pts < sePtsNumber; ++pts) {                                    \
    b.x = a.x + se.points[pts].x;                                              \
    b.y = a.y + se.points[pts].y;                                              \
    b.z = a.z + se.points[pts].z;                                              \
    if (b.x < S[0] && b.y < S[1] && b.z < S[2]) {                              \
      CoorToIndex(b);
#define ENDForEachNeighborOf                                                   \
  }                                                                            \
  }

#define LIVEVERSION

  template <class T1, class T2>
  void geoDistance(const Image<T1> &_in_, Image<T2> &_out_, const StrElt &se)
  {
    queue<size_t> *c1 = new queue<size_t>(), *c2 = new queue<size_t>(), *tmp;
    T1            *in  = _in_.getPixels();
    T2            *out = _out_.getPixels();
    index_T        p, q;

    UINT   sePtsNumber = se.points.size();
    UINT   pts;
    size_t S[3];
    _in_.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];

    ForEachPixel(p)
    {
      if (out[p.o] == 1) {
        c2->push(p.o);
      }
    }
    ENDForEachPixel

        T2 d = 1;
    while (!c2->empty()) {
      tmp = c1;
      c1  = c2;
      c2  = tmp;
      ++d;

      while (!c1->empty()) {
        p.o = c1->front();
        c1->pop();
        IndexToCoor(p);

        ForEachNeighborOf(p, q)
        {
          if (in[p.o] == in[q.o] && out[q.o] == 0) {
            out[q.o] = d;
            c2->push(q.o);
          }
        }
        ENDForEachNeighborOf
      }
    }

    delete c1;
    delete c2;
  }

} // namespace smil

#endif // _D_UTILS_H_
