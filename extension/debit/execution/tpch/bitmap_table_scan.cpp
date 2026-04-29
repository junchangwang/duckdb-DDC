#include "execution/tpch/bitmap_table_scan.hpp"

namespace duckdb {

BMTableScan::BMTableScan()
    : row_ids(new vector<row_t>()),
      cursor(new idx_t(0)),
      num_idlist(0) {}

BMTableScan::~BMTableScan() {
    delete row_ids;
    delete cursor;
}

}
