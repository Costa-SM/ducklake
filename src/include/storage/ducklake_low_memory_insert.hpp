//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_low_memory_insert.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "storage/ducklake_insert.hpp"

namespace duckdb {

class DuckLakeTableEntry;
class DuckLakeCatalog;

//===--------------------------------------------------------------------===//
// DuckLakeLowMemoryInsert
//===--------------------------------------------------------------------===//
// A physical operator that implements radix hierarchical insert for
// high-cardinality partitioned tables. Instead of processing all partitions
// at once (which can OOM), it processes partitions in groups (super-partitions).
//
// Strategy:
// 1. Buffer incoming data
// 2. When buffer is full, partition data into super-partition groups
// 3. For each super-partition group, execute a filtered insert
// 4. This limits concurrent partition count to ~(total_partitions / num_super_partitions)

class DuckLakeLowMemoryInsertGlobalState : public GlobalSinkState {
public:
	explicit DuckLakeLowMemoryInsertGlobalState(ClientContext &context, DuckLakeTableEntry &table,
	                                            const vector<LogicalType> &types);

	DuckLakeTableEntry &table;
	//! Buffered data waiting to be processed
	unique_ptr<ColumnDataCollection> buffered_data;
	//! Total rows inserted across all batches
	idx_t total_insert_count = 0;
	//! All written files from all batches
	vector<DuckLakeDataFile> written_files;
	//! Partition column index in the input
	idx_t partition_column_idx;
	//! Number of super-partitions to use
	idx_t num_super_partitions;
	//! Maximum rows to buffer before processing
	idx_t buffer_threshold;
	//! Mutex for thread safety
	mutex lock;
};

class DuckLakeLowMemoryInsert : public PhysicalOperator {
public:
	static constexpr idx_t DEFAULT_BUFFER_THRESHOLD = 1000000;  // 1M rows

	DuckLakeLowMemoryInsert(PhysicalPlan &physical_plan, const vector<LogicalType> &types,
	                        DuckLakeTableEntry &table, idx_t partition_column_idx,
	                        string encryption_key, idx_t estimated_partition_count);

	//! The table to insert into
	DuckLakeTableEntry &table;
	//! Index of the partition column in input
	idx_t partition_column_idx;
	//! Encryption key for Parquet files
	string encryption_key;
	//! Estimated number of partitions
	idx_t estimated_partition_count;

public:
	// Source interface
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

public:
	// Sink interface
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;  // Multiple threads can append to buffer
	}

	string GetName() const override {
		return "DUCKLAKE_LOW_MEMORY_INSERT";
	}

private:
	//! Process the buffered data in super-partition batches
	void ProcessBufferedData(ClientContext &context, DuckLakeLowMemoryInsertGlobalState &gstate) const;
	//! Execute insert for a specific super-partition range
	void InsertSuperPartition(ClientContext &context, DuckLakeLowMemoryInsertGlobalState &gstate,
	                          ColumnDataCollection &data, idx_t super_partition_idx) const;
};

} // namespace duckdb

