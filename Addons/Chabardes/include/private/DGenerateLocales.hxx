#ifndef _GENERATELOCALES_HPP_
#define _GENERATELOCALES_HPP_

namespace smil
{
#define localesT vector<vector<UINT>>

  double radius_dist(IntPoint p, IntPoint q)
  {
    //    return sqrt (pow (p.x - q.x,2) + pow (p.y - q.y,2) + pow (p.z -
    //    q.z,2));

    double pmag = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    double px   = p.x / pmag;
    double py   = p.y / pmag;
    double pz   = p.z / pmag;
    double qmag = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    double qx   = q.x / qmag;
    double qy   = q.y / qmag;
    double qz   = q.z / qmag;

    return (double) (acos(px * qx + py * qy + pz * qz));
  }

  void generateNeighbors(localesT &neighbors, const StrElt &s)
  {
    StrElt se = s.noCenter();
    neighbors = localesT(2 * se.points.size(), vector<UINT>());

    UINT a[se.points.size() * se.points.size()] = {0};
    UINT index                                  = se.points.size();

    double dist;

    for (UINT i = 0; i < se.points.size(); ++i) {
      dist = 360.0;
      for (UINT j = 0; j < se.points.size(); ++j) {
        if (j != i && radius_dist(se.points[i], se.points[j]) < dist) {
          dist = radius_dist(se.points[i], se.points[j]);
        }
      }
      for (UINT j = 0; j < se.points.size(); ++j) {
        if (j != i && dist == radius_dist(se.points[i], se.points[j])) {
          if (a[min(i, j) * se.points.size() + max(i, j)] == 0)
            a[min(i, j) * se.points.size() + max(i, j)] = index++;
          neighbors[i].push_back(a[min(i, j) * se.points.size() + max(i, j)]);
          neighbors[a[min(i, j) * se.points.size() + max(i, j)]].push_back(i);
        }
      }
    }
  }

  void generateReachables(localesT &reachables, const localesT &neighbors,
                          const StrElt &s)
  {
    StrElt se  = s.noCenter();
    reachables = localesT(2 * se.points.size(), vector<UINT>());

    double dist;

    for (UINT i = 0; i < se.points.size(); ++i) {
      dist = 0.0;
      for (UINT j = 0; j < se.points.size(); ++j) {
        if (j != i && radius_dist(se.points[i], se.points[j]) > dist) {
          dist = radius_dist(se.points[i], se.points[j]);
        }
      }

      for (UINT j = 0; j < se.points.size(); ++j) {
        if (j != i && dist == radius_dist(se.points[i], se.points[j])) {
          reachables[i].push_back(j);
        }
      }
    }

    for (UINT i = se.points.size(); i < 2 * se.points.size(); ++i) {
      vector<UINT> v = vector<UINT>(se.points.size() * 2, 0);
      for (UINT j = 0; j < neighbors[i].size(); ++j) {
        UINT a = reachables[neighbors[i][j]][0];
        for (UINT k = 0; k < neighbors[a].size(); ++k) {
          v[neighbors[a][k]]++;
        }
      }
      for (UINT j = 0; j < se.points.size() * 2; ++j)
        if (v[j] == 1)
          reachables[i].push_back(j);
    }
  }

  void generateInverses(vector<UINT> &inverses, const StrElt &s)
  {
    StrElt se  = s.noCenter();
    StrElt tse = se.transpose();
    inverses   = vector<UINT>(se.points.size(), 0);

    for (UINT i = 0; i < se.points.size(); ++i)
      for (UINT j = 0; j < se.points.size(); ++j)
        if (se.points[i].x == tse.points[j].x &&
            se.points[i].y == tse.points[j].y &&
            se.points[i].z == tse.points[j].z) {
          inverses[i] = j;
        }
  }

} // namespace smil

#endif // _GENERATELOCALES_HPP_
