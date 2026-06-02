#pragma once

#include "duckdb.hpp"
#ifndef DUCKDB_AMALGAMATION
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#endif

namespace duckdb {
class ClientContext;
}

namespace bmtpch {

struct DBGenWrapper {

	static std::string GetQuery(int query); // TPC-H query text
};

}
