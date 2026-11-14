// The frontend: reads the exporter's graph.json and lowers ATen operators to
// the graph IR.
#pragma once

#include <string>

#include "graph/ir.h"
#include "support/json.h"

namespace mlc {

struct ImportError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ImportedModel {
  Graph graph;
  std::string weights_file;  // as written in graph.json, relative to it
  int64_t weights_bytes = 0;
  int aten_nodes = 0;        // how many exported nodes were lowered
};

// Lowers a parsed graph.json. Every exported node must be understood: an
// operator with no lowering is an ImportError naming it, never a silent skip.
// The returned graph has passed the verifier.
ImportedModel import_graph(const json::Value& doc);

ImportedModel import_graph_file(const std::string& path);

}  // namespace mlc
