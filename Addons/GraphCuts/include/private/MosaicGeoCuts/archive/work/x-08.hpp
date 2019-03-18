template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
          typename _Sigma, class SE, class ImageOut>
RES_T MAP_MRF_Ising(const ImageIn &imIn, const ImageMosaic &imMosaic,
                      const ImageMarker &imMarker, const _Beta Beta,
                      const _Sigma Sigma, const SE &nl, ImageOut &imOut)
{
  MORPHEE_ENTER_FUNCTION("t_MAP_MRF_Ising");

  std::cout << "Enter function t_MAP_MRF_Ising" << std::endl;

  if ((!imOut.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imIn.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imMosaic.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imMarker.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  // common image iterator

  typename ImageIn::const_iterator it, iend;
  morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
  typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
  offset_t o0;
  offset_t o1;

  // needed for max flow: capacit map, rev_capacity map, etc.
  typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                       boost::directedS>
      Traits;
  typedef boost::adjacency_list<
      boost::listS, boost::vecS, boost::directedS,
      boost::property<boost::vertex_name_t, std::string>,
      boost::property<
          boost::edge_capacity_t, double,
          boost::property<
              boost::edge_residual_capacity_t, double,
              boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>
      Graph_d;

  Graph_d g;

  double sigma = Sigma;

  boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
      boost::get(boost::edge_capacity, g);

  boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
      get(boost::edge_reverse, g);

  boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
      residual_capacity = get(boost::edge_residual_capacity, g);

  bool in1;
  Graph_d::edge_descriptor e1, e2, e3, e4;
  Graph_d::vertex_descriptor vSource, vSink;
  int numVert = 1;

  std::cout << "build Region Adjacency Graph" << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex
    o0      = it.getOffset();
    int val = imMosaic.pixelFromOffset(o0);
    imOut.setPixel(o0, 0);

    if (val > numVert) {
      numVert = val;
    }
  }

  double *mean = new double[numVert + 1];
  int *nb_val  = new int[numVert + 1];
  int *marker  = new int[numVert + 1];

  for (int i = 0; i <= numVert; i++) {
    boost::add_vertex(g);
    mean[i]   = 0;
    nb_val[i] = 0;
    marker[i] = 0;
  }

  vSource = boost::add_vertex(g);
  vSink   = boost::add_vertex(g);

  double meanforeground = 0;
  double meanbackground = 0;
  double nb_foreground  = 0;
  double nb_background  = 0;

  std::cout << "Compute Mean Value in Regions" << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex
    o0            = it.getOffset();
    int val       = imIn.pixelFromOffset(o0);
    int val2      = imMosaic.pixelFromOffset(o0);
    int val3      = imMarker.pixelFromOffset(o0);
    double valeur = (double) val;

    mean[val2]   = mean[val2] + (valeur / 255.0);
    nb_val[val2] = nb_val[val2] + 1;

    if (val3 == 2) {
      meanforeground = meanforeground + (valeur / 255.0);
      nb_foreground++;
      marker[val2] = 2;
    } else if (val3 == 3) {
      meanbackground = meanbackground + (valeur / 255.0);
      nb_background++;
      marker[val2] = 3;
    }
  }

  meanforeground = meanforeground / nb_foreground;
  meanbackground = meanbackground / nb_background;

  std::cout << "Mean Foreground " << meanforeground << std::endl;
  std::cout << "Mean Background " << meanbackground << std::endl;

  std::cout << "Compute terminal links" << std::endl;

  double sigmab = 0.2;
  sigmab        = Sigma;

  for (int i = 0; i <= numVert; i++) {
    mean[i] = mean[i] / (nb_val[i]);

    if (marker[i] == 2 && nb_val[i] > 0) {
      boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
      boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
      capacity[e4]        = (std::numeric_limits<double>::max)();
      capacity[e3]        = (std::numeric_limits<double>::max)();
      rev[e4]             = e3;
      rev[e3]             = e4;
    } else if (marker[i] == 3 && nb_val[i] > 0) {
      boost::tie(e4, in1) = boost::add_edge(vSink, i, g);
      boost::tie(e3, in1) = boost::add_edge(i, vSink, g);
      capacity[e4]        = (std::numeric_limits<double>::max)();
      capacity[e3]        = (std::numeric_limits<double>::max)();
      rev[e4]             = e3;
      rev[e3]             = e4;

    } else if (nb_val[i] > 0) {
      double valee = mean[i];
      double sigma = Sigma;
      double val2  = (valee - meanforeground) * (valee - meanforeground) /
                    (2 * sigma * sigma);
      double val1 = (valee - meanbackground) * (valee - meanbackground) /
                    (2 * sigmab * sigmab);

      boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
      boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
      capacity[e4]        = nb_val[i] * val1;
      capacity[e3]        = nb_val[i] * val1;
      rev[e4]             = e3;
      rev[e3]             = e4;

      boost::tie(e4, in1) = boost::add_edge(i, vSink, g);
      boost::tie(e3, in1) = boost::add_edge(vSink, i, g);
      capacity[e4]        = nb_val[i] * val2;
      capacity[e3]        = nb_val[i] * val2;
      rev[e4]             = e3;
      rev[e3]             = e4;
    }
  }

  delete[] nb_val;
  delete[] mean;
  int numEdges = 0;

  std::cout << "build Region Adjacency Graph Edges" << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex and an edge
    o1      = it.getOffset();
    int val = imMosaic.pixelFromOffset(o1);
    neighb.setCenter(o1);

    for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
      const offset_t o2 = nit.getOffset();
      int val2          = imMosaic.pixelFromOffset(o2);

      if (o2 > o1) {
        if (val != val2) {
          boost::tie(e3, in1) = boost::edge(val, val2, g);
          double cost         = (double) Beta;
          if (in1 == 0) {
            numEdges++;
            boost::tie(e4, in1) = boost::add_edge(val, val2, g);
            boost::tie(e3, in1) = boost::add_edge(val2, val, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          } else {
            boost::tie(e4, in1) = boost::edge(val, val2, g);
            boost::tie(e3, in1) = boost::edge(val2, val, g);
            capacity[e4]        = capacity[e4] + cost;
            capacity[e3]        = capacity[e3] + cost;
          }
        }
      }
    }
  }

  std::cout << "Number of vertices " << numVert << std::endl;
  std::cout << "Number of Edges " << numEdges << std::endl;

  boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
      boost::get(boost::vertex_index, g);
  std::vector<boost::default_color_type> color(boost::num_vertices(g));

  std::cout << "Compute Max flow" << std::endl;
#if BOOST_VERSION >= 104700
  double flow = boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                           &color[0], indexmap, vSource, vSink);
#else
  double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                    &color[0], indexmap, vSource, vSink);
#endif
  std::cout << "c  The total flow:" << std::endl;
  std::cout << "s " << flow << std::endl << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex and an edge
    o1      = it.getOffset();
    int val = imMosaic.pixelFromOffset(o1);

    if (color[val] == color[vSource])
      imOut.setPixel(o1, 2);
    else if (color[val] == color[vSink])
      imOut.setPixel(o1, 3);
    else
      imOut.setPixel(o1, 4);
  }

  return RES_OK;
}
