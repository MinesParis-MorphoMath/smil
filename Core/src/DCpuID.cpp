/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "Core/include/DCpuID.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <regex>
#include <iostream>

using namespace smil;

CpuID::CpuID()
{
  cores         = 0;
  logical       = 0;
  hyperThreaded = true;
#ifdef USE_OPEN_MP
  // #pragma omp parallel
  {
    cores   = omp_get_num_procs();
    logical = omp_get_max_threads();
  }
  // if (hyperThreaded)
  //   cores /= 2;
#endif

#if defined(__linux__)
  std::ifstream fin;
  fin = std::ifstream("/proc/cpuinfo");
  if (fin.good()) {
    int         nprocs = 0;
    std::string svendor;
    std::string smodel;
    std::string buffer;

    char line[1024];
    while (fin.getline(line, sizeof line)) {
      std::string sline = std::string(line);

      if (_get_value(sline, "processor", buffer))
        nprocs++;
      if (_get_value(sline, "model name", buffer))
        this->model = buffer;
      if (_get_value(sline, "vendor_id", buffer))
        this->vendor = buffer;
      if (_get_value(sline, "flags", buffer))
        this->flags = buffer;
    }
    if (cores == 0 || logical == 0)
      cores = logical = nprocs;
  }
#endif // __linux__

  cores   = std::max(cores, 4U);
  logical = std::max(logical, 4U);
}

bool CpuID::_get_value(std::string &s, const char *prefix, std::string &value)
{
  bool        ok = false;
  std::regex  re(std::string("^") + std::string(prefix) +
                 "[[:space:]]*:[[:space:]]*");
  std::smatch sm;
  if (regex_search(s, sm, re)) {
    value = sm.suffix();
    ok    = true;
  }
  return ok;
}
