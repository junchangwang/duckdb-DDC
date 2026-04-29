//===----------------------------------------------------------------------===//
//                         DuckDB
//
// debit_extension.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"

namespace duckdb {

class DebitExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
	std::string Version() const override;

	static std::string GetQuery(int query);
};

} // namespace duckdb
