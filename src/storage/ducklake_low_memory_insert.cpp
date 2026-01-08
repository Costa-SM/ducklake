#include "storage/ducklake_low_memory_insert.hpp"
#include "storage/ducklake_catalog.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_schema_entry.hpp"
#include "storage/ducklake_transaction.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Global State
//===--------------------------------------------------------------------===//
DuckLakeLowMemoryInsertGlobalState::DuckLakeLowMemoryInsertGlobalState(ClientContext &context,
                                                                        DuckLakeTableEntry &table,
                                                                        const vector<LogicalType> &types)
    : table(table) {
	buffered_data = make_uniq<ColumnDataCollection>(context, types);

	// Calculate number of super-partitions based on RadixInsertConfig
	num_super_partitions = RadixInsertConfig::NUM_SUPER_PARTITIONS;
	buffer_threshold = DuckLakeLowMemoryInsert::DEFAULT_BUFFER_THRESHOLD;
}

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//
DuckLakeLowMemoryInsert::DuckLakeLowMemoryInsert(PhysicalPlan &physical_plan, const vector<LogicalType> &types,
                                                  DuckLakeTableEntry &table_p, idx_t partition_column_idx_p,
                                                  string encryption_key_p, idx_t estimated_partition_count_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types, 1), table(table_p),
      partition_column_idx(partition_column_idx_p), encryption_key(std::move(encryption_key_p)),
      estimated_partition_count(estimated_partition_count_p) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
unique_ptr<GlobalSinkState> DuckLakeLowMemoryInsert::GetGlobalSinkState(ClientContext &context) const {
	// Get the input types from children
	vector<LogicalType> input_types;
	if (!children.empty()) {
		input_types = children[0].get().types;
	}
	return make_uniq<DuckLakeLowMemoryInsertGlobalState>(context, table, input_types);
}

SinkResultType DuckLakeLowMemoryInsert::Sink(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<DuckLakeLowMemoryInsertGlobalState>();

	// Append to buffer (thread-safe)
	{
		lock_guard<mutex> guard(gstate.lock);
		gstate.buffered_data->Append(chunk);
	}

	// Check if we should process the buffer
	if (gstate.buffered_data->Count() >= gstate.buffer_threshold) {
		// Process in a thread-safe manner
		lock_guard<mutex> guard(gstate.lock);
		if (gstate.buffered_data->Count() >= gstate.buffer_threshold) {
			ProcessBufferedData(context.client, gstate);
		}
	}

	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType DuckLakeLowMemoryInsert::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                    OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<DuckLakeLowMemoryInsertGlobalState>();

	// Process any remaining buffered data
	if (gstate.buffered_data->Count() > 0) {
		ProcessBufferedData(context, gstate);
	}

	// Calculate total insert count
	for (auto &data_file : gstate.written_files) {
		gstate.total_insert_count += data_file.row_count;
	}

	// Commit the written files to the transaction
	auto &transaction = DuckLakeTransaction::Get(context, table.catalog);
	transaction.AppendFiles(table.GetTableId(), std::move(gstate.written_files));

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// GetData (Source)
//===--------------------------------------------------------------------===//
SourceResultType DuckLakeLowMemoryInsert::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                           OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<DuckLakeLowMemoryInsertGlobalState>();
	auto value = Value::BIGINT(NumericCast<int64_t>(gstate.total_insert_count));
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, value);
	return SourceResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Process Buffered Data
//===--------------------------------------------------------------------===//
void DuckLakeLowMemoryInsert::ProcessBufferedData(ClientContext &context,
                                                   DuckLakeLowMemoryInsertGlobalState &gstate) const {
	if (gstate.buffered_data->Count() == 0) {
		return;
	}

	// Get partition column index
	idx_t part_col_idx = partition_column_idx;

	// Create super-partition collections
	idx_t num_super = gstate.num_super_partitions;
	vector<unique_ptr<ColumnDataCollection>> super_partitions;
	super_partitions.resize(num_super);
	for (idx_t i = 0; i < num_super; i++) {
		super_partitions[i] = make_uniq<ColumnDataCollection>(context, gstate.buffered_data->Types());
	}

	// Scan buffered data and distribute to super-partitions
	DataChunk scan_chunk;
	scan_chunk.Initialize(context, gstate.buffered_data->Types());

	ColumnDataScanState scan_state;
	gstate.buffered_data->InitializeScan(scan_state);

	// Calculate partitions per super-partition
	idx_t partitions_per_super = RadixInsertConfig::GetPartitionsPerSuperPartition(
	    estimated_partition_count, num_super);

	while (gstate.buffered_data->Scan(scan_state, scan_chunk)) {
		// Get partition column values
		auto &partition_col = scan_chunk.data[part_col_idx];
		UnifiedVectorFormat partition_data;
		partition_col.ToUnifiedFormat(scan_chunk.size(), partition_data);

		// Create selection vectors for each super-partition
		vector<SelectionVector> selections(num_super);
		vector<idx_t> counts(num_super, 0);
		for (idx_t i = 0; i < num_super; i++) {
			selections[i].Initialize(STANDARD_VECTOR_SIZE);
		}

		// Distribute rows to super-partitions based on partition value
		auto partition_values = UnifiedVectorFormat::GetData<int64_t>(partition_data);
		for (idx_t row = 0; row < scan_chunk.size(); row++) {
			idx_t idx = partition_data.sel->get_index(row);
			int64_t partition_value = partition_values[idx];

			// Map partition value to super-partition
			idx_t super_idx;
			if (partitions_per_super > 0) {
				super_idx = static_cast<idx_t>(partition_value) / partitions_per_super;
				super_idx = MinValue(super_idx, num_super - 1);
			} else {
				super_idx = static_cast<idx_t>(partition_value) % num_super;
			}

			selections[super_idx].set_index(counts[super_idx]++, row);
		}

		// Append to each super-partition's collection
		for (idx_t i = 0; i < num_super; i++) {
			if (counts[i] > 0) {
				DataChunk sliced;
				sliced.Initialize(context, scan_chunk.GetTypes());
				sliced.Slice(scan_chunk, selections[i], counts[i]);
				super_partitions[i]->Append(sliced);
			}
		}

		scan_chunk.Reset();
	}

	// Process each super-partition sequentially
	// This limits the number of concurrent partitions in DuckDB
	for (idx_t super_idx = 0; super_idx < num_super; super_idx++) {
		if (super_partitions[super_idx]->Count() > 0) {
			InsertSuperPartition(context, gstate, *super_partitions[super_idx], super_idx);
			// Clear to free memory before processing next super-partition
			super_partitions[super_idx].reset();
		}
	}

	// Clear the main buffer
	gstate.buffered_data->Reset();
}

//===--------------------------------------------------------------------===//
// Insert Super-Partition
//===--------------------------------------------------------------------===//
void DuckLakeLowMemoryInsert::InsertSuperPartition(ClientContext &context,
                                                    DuckLakeLowMemoryInsertGlobalState &gstate,
                                                    ColumnDataCollection &data,
                                                    idx_t super_partition_idx) const {
	// This super-partition contains only a subset of partition values.
	// When we insert this data, DuckDB's PhysicalCopyToFile will only see
	// a limited number of unique partition values (~partitions/256).

	if (data.Count() == 0) {
		return;
	}

	// Convert ColumnDataCollection to vector<vector<Value>> for ValueRelation
	vector<vector<Value>> values;
	values.reserve(data.Count());

	DataChunk scan_chunk;
	scan_chunk.Initialize(context, data.Types());

	ColumnDataScanState scan_state;
	data.InitializeScan(scan_state);

	while (data.Scan(scan_state, scan_chunk)) {
		for (idx_t row = 0; row < scan_chunk.size(); row++) {
			vector<Value> row_values;
			row_values.reserve(scan_chunk.ColumnCount());
			for (idx_t col = 0; col < scan_chunk.ColumnCount(); col++) {
				row_values.push_back(scan_chunk.GetValue(col, row));
			}
			values.push_back(std::move(row_values));
		}
		scan_chunk.Reset();
	}

	if (values.empty()) {
		return;
	}

	// Get column names from the table
	vector<string> column_names;
	for (auto &col : table.GetColumns().Physical()) {
		column_names.push_back(col.Name());
	}

	// Create a ValueRelation from the data
	auto value_relation = make_shared_ptr<ValueRelation>(
	    context.shared_from_this(),
	    values,
	    column_names,
	    "super_partition_" + to_string(super_partition_idx)
	);

	// Get the target table info
	auto &catalog = table.ParentCatalog();
	auto &schema = table.ParentSchema();

	// Execute the INSERT using ValueRelation
	// This will go through the standard DuckLake insert path, but with only
	// a subset of partition values (so ~partitions/256 active partitions)
	value_relation->Insert(catalog.GetName(), schema.name, table.name);

	// Note: The written files are captured by the DuckLake transaction automatically.
	// We don't need to manually track them here because the INSERT above goes through
	// the standard DuckLakeCatalog::PlanInsert path, which creates a DuckLakeInsert
	// operator that adds files to the transaction.
}

} // namespace duckdb

