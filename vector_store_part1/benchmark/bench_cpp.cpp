// bench_cpp.cpp
//
// WHAT THIS PROGRAM IS FOR
// ------------------------
// This measures how fast our vector store answers nearest-neighbour searches.
//
// THE PLAN
//   1. Load the stored vectors from the file gen_data made.
//   2. Load the query vectors from the other file.
//   3. Run a few queries WITHOUT timing them (warming up the cache).
//   4. Run every query again, this time with a stopwatch around the search.
//   5. Print how long each search took on average.
//
// We only care about the search step time.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

// The store under test. Which implementation this resolves to is chosen at
// compile time by the -I (e.g.
// -I../impls/initial_design_no_opt), so the SAME source benchmarks any impls/<keyword>.
#include "vector_store.h"

// ENGINE_LABEL will be passed from bench.py
#if defined(ENGINE_LABEL)
const char* ENGINE_NAME = ENGINE_LABEL;
#else
const char* ENGINE_NAME = "unknown";
#endif

// Map a metric name (from the command line) to the store's Metric enum.
// Accepts "euclidean"" and "cosine"; anything else falls back to Euclidean.
Metric parse_metric(const std::string& name) {
    if (name == "cosine") return Metric::Cosine;
    return Metric::Euclidean;  // "euclidean", "l2", or unset
}

// Read the query file into a list of vectors.
// The file format is "id v0 v1 v2 ..." per line (same as the vectors file).
// We do read the query id but only the vector values matter.
std::vector<std::vector<double>> read_queries(const std::string& path) {
    std::vector<std::vector<double>> queries;

    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);

        int id;
        if (!(stream >> id)) {
            continue;  // skip blank or broken lines
        }

        // Read the rest of the numbers on the line into one query vector.
        std::vector<double> values;
        double value;
        while (stream >> value) {
            values.push_back(value);
        }
        queries.push_back(values);
    }
    return queries;
}

int main(int argc, char** argv) {
    // We need to know: which vectors file, what dimension, and how many
    // neighbours (k) to return.
    if (argc < 4) {
        std::cout << "usage: " << argv[0]
                  << " <vectors_file> <dim> <k> [metric]\n";
        return 1;
    }
    std::string vectors_path = argv[1];
    int dim = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);
    // Optional 4th arg picks the distance metric (default Euclidean).
    Metric metric = parse_metric(argc >= 5 ? argv[4] : "euclidean");

    // The query file name only depends on the dimension (see gen_data.cpp).
    std::string queries_path = "queries_" + std::to_string(dim) + ".txt";

    // --- Step 1: load the stored vectors ---
    VectorStore store(dim);
    auto load_start = std::chrono::steady_clock::now();
    if (!store.load(vectors_path)) {
        std::cout << "Could not open " << vectors_path << "\n";
        return 1;
    }
    auto load_end = std::chrono::steady_clock::now();

    // --- Step 2: load the queries. ---
    std::vector<std::vector<double>> queries = read_queries(queries_path);
    if (queries.empty()) {
        std::cout << "No queries found in " << queries_path << "\n";
        return 1;
    }

    // --- Step 3: warm-up (not timed). ---
    // The very first searches are just for warming up the caches.
    // The checksum is for a quick sanity check between runs and to 
    // avoid any potential dead code optimization from compiler.
    long long checksum = 0;
    for (int i = 0; i < 3 && i < int(queries.size()); ++i) {
        std::vector<Result> results = store.search(queries[i], k, metric);
        if (!results.empty()) {
            checksum += results[0].id;
        }
    }

    // --- Step 4: the timed run. ---
    // We search with every query and add up the total time.
    auto search_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        std::vector<Result> results =
            store.search(queries[i], k, metric);
        if (!results.empty()) {
            checksum += results[0].id;
        }
    }
    auto search_end = std::chrono::steady_clock::now();

    // --- Step 5: Calculate per query average ---
    double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();
    double search_ms =
        std::chrono::duration<double, std::milli>(search_end - search_start)
            .count();
    double per_query_ms = search_ms / queries.size();
    double queries_per_sec = 1000.0 / per_query_ms;

    std::cout << "engine=" << ENGINE_NAME
              << "  N=" << store.size()
              << "  dim=" << dim
              << "  k=" << k
              << "  queries=" << queries.size()
              << "  load_ms=" << load_ms
              << "  per_query_ms=" << per_query_ms
              << "  queries_per_sec=" << queries_per_sec
              << "  (checksum=" << checksum << ")\n";
    return 0;
}
