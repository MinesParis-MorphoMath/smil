template <class ImageIn, class ImageMosaic, class ImageMarker, class SE,
          class ImageOut>
RES_T GeoCuts_Segment_Graph(const ImageIn &imIn, const ImageMosaic &imMosaic,
                              const ImageMarker &imMarker, const SE &nl,
                              ImageOut &imOut)
{
  MORPHEE_ENTER_FUNCTION("t_GeoCuts_Segment_Graph");

  std::cout << "Enter function optimize mosaic t_GeoCuts_Segment_Graph"
            << std::endl;

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

  double sigma = 1.0;

  boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
      boost::get(boost::edge_capacity, g);

  boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
      get(boost::edge_reverse, g);

  boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
      residual_capacity = get(boost::edge_residual_capacity, g);

  bool in1;
  Graph_d::edge_descriptor e1, e2, e3, e4;
  Graph_d::vertex_descriptor vSource, vSink;
  int numVert         = 0;
  double meanclasse1  = 0.0;
  double meanclasse12 = 0.0;

  double meanclasse2 = 0.5;

  double sigma1 = 0.25;
  double sigma2 = 0.5;

  double max_value    = 0.0;
  double max_longueur = 0.0;

  std::cout << "build Region Adjacency Graph" << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex
    o0       = it.getOffset();
    int val  = imMosaic.pixelFromOffset(o0);
    int val2 = imIn.pixelFromOffset(o0);
    int val3 = imMarker.pixelFromOffset(o0);

    if (val > numVert) {
      numVert = val;
    }
    if (val2 > max_value) {
      max_value = val2;
    }
    if (val3 > max_longueur) {
      max_longueur = val3;
    }
  }

  std::cout << "build Region Adjacency Graph Vertices" << std::endl;

  std::cout << "Number of Vertices : " << numVert << std::endl;

  std::cout << "Max value : " << max_value << std::endl;

  for (int i = 0; i <= numVert; i++) {
    boost::add_vertex(g);
  }

  vSource = boost::add_vertex(g);
  vSink   = boost::add_vertex(g);

  std::cout << "build Region Adjacency Graph Edges" << std::endl;

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    // for all pixels in imIn create a vertex and an edge
    o1      = it.getOffset();
    int val = imMosaic.pixelFromOffset(o1);

    if (val > 0) {
      boost::tie(e4, in1) = boost::edge(vSource, val, g);

      if (in1 == 0) {
        // std::cout<<"Add new edge marker 2"<<std::endl;
        double valee    = (double) imIn.pixelFromOffset(o1) / max_value;
        double longueur = (double) imMarker.pixelFromOffset(o1) / max_longueur;

        double cost1 = 4 * (1 - longueur) + (valee - meanclasse1) *
                                                (valee - meanclasse1) /
                                                (2 * sigma1 * sigma1);
        double cost12 = 4 * (1 - longueur) + (valee - meanclasse12) *
                                                 (valee - meanclasse12) /
                                                 (2 * sigma1 * sigma1);
        double cost2 = 4 * (1 - 0.17) + (valee - meanclasse2) *
                                            (valee - meanclasse2) /
                                            (2 * sigma2 * sigma2);

        /*
          double cost1 =
          (valee-meanclasse1)*(valee-meanclasse1)/(2*sigma1*sigma1); double
          cost12 =
          (valee-meanclasse12)*(valee-meanclasse12)/(2*sigma1*sigma1);
          double cost2 =
          (valee-meanclasse2)*(valee-meanclasse2)/(2*sigma2*sigma2);
        */

        boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
        boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
        capacity[e4]        = std::min(cost1, cost12);
        capacity[e3]        = std::min(cost1, cost12);
        rev[e4]             = e3;
        rev[e3]             = e4;

        boost::tie(e4, in1) = boost::add_edge(vSink, val, g);
        boost::tie(e3, in1) = boost::add_edge(val, vSink, g);
        capacity[e4]        = cost2;
        capacity[e3]        = cost2;
        rev[e4]             = e3;
        rev[e3]             = e4;
      }

      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imMosaic.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2 && val2 > 0) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);

            double valee1 = (double) imIn.pixelFromOffset(o1) / max_value;
            double valee2 = (double) imIn.pixelFromOffset(o2) / max_value;

            double longueur1 =
                (double) imMarker.pixelFromOffset(o1) / max_longueur;
            double longueur2 =
                (double) imMarker.pixelFromOffset(o2) / max_longueur;

            double cost_diff =
                0.01 * std::exp(-0.01 * (valee1 - valee2) * (valee1 - valee2));
            double cost_longueur = 0.1;

            if (in1 == 0) {
              // std::cout<<"Add new edge"<<std::endl;
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost_longueur;
              capacity[e3]        = cost_longueur;
              rev[e4]             = e3;
              rev[e3]             = e4;
            }
          }
        }
      }
    }
  }

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
      imOut.setPixel(o1, 10);
    if (color[val] == 1)
      imOut.setPixel(o1, 0);
    if (color[val] == color[vSink])
      imOut.setPixel(o1, 30);
  }

  return RES_OK;
}
