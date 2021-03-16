/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021, Centre de Morphologie Mathematique
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   Common Geo-Cuts Tools
 *
 * History :
 *   - 20/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it...
 *   - XX/XX/XXXX -
 *
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_GEOCUTSTOOLS_HPP_
#define _D_GEOCUTSTOOLS_HPP_

namespace geocuts
{
  /*
   * Some useful tools
   */

/*
 *
 *
 *
 */
#ifndef __BOOST_INCLUDED__
#define __BOOST_INCLUDED__

/*
 *To prevent doxygen try to document boost included headers
 */
/** @cond */
#include <boost/config.hpp>
// for boost::tie
#include <boost/utility.hpp>
// for boost::graph_traits
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#endif

/** @endcond */
#endif //  __BOOST_INCLUDED__

  /** @cond */

  static int debugOn = false;

  /*
  typedef struct {
    float x;
    float y;
    float p;
  } morceau;

  typedef std::list<morceau> affine_par_morceaux;
  */

#ifndef SMIL_ENTER_FUNCTION
#define SMIL_ENTER_FUNCTION(a)                                                 \
  {                                                                            \
    if (debugOn)                                                               \
      std::cout << "Entering function " << __func__ << " " << (a)              \
                << std::endl;                                                  \
  }

#define SMIL_REGISTER_ERROR(a)
#endif

#define M_PI 3.14159265358979323846
#define INFINI_POSITIF std::numeric_limits<double>::max)()
#define _SECURE_SCL 0

  /* This function selects points from an Structuring element
   * generating a positive offset
   */
  inline vector<IntPoint> filterStrElt(StrElt se)
  {
    vector<IntPoint> pts(0);

    vector<IntPoint>::iterator it, itStart, itEnd;
    itStart = se.points.begin();
    itEnd   = se.points.end();

    for (it = itStart; it != itEnd; it++) {
      bool ok = (4 * it->z + 2 * it->y + it->x) > 0;
      if (ok)
        pts.push_back(*it);

      if (debugOn) {
        std::cout << (ok ? "GOOD " : "BAD  ") << std::right << " "
                  << std::setw(6) << it->x << " " << std::setw(6) << it->y
                  << " " << std::setw(6) << it->z << endl;
      }
    }
    return pts;
  }

#if 0
  typedef graph_traits<Graph>::vertex_descriptor vertexDescriptor_T;

  inline kolmogorovMaxFlow(Graph& g,
                           CapacityEdgeMap capacity,
                           ResidualCapacityEdgeMap residual_capacity,
                           ReverseEdgeMap rev,
                           ColorMap color,
                           IndexMap indexmap,
                           vertexDescriptor_T vSource,
                           vertexDescriptor_T vSink);
  {
    double flow;

#if BOOST_VERSION >= 104700
    flow = boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                   color, indexmap, vSource, vSink);
#else
    flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                      color, indexmap, vSource, vSink);
#endif
    return flow;    
  }                                   
#endif

  /** @endcond */
} // namespace geocuts

#endif // _D_GEOCUTSTOOLS_HPP_

