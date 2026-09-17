// bench_sqlite.cpp
//
// This measures how fast sqlite-vec's
// "vec0" does exactly the same job our store does: given a query vector, find
// the closest stored vectors.
//
//
// THE PLAN (mirrors bench_cpp.cpp on purpose)
//   1. Turn on vec0 and make an in-memory database.
//   2. Create the vec0 table and INSERT the stored vectors.
//   3. Read the query vectors from the same queries file.
//   4. Warm up with a few queries.
//   5. Run every query with the stopwatch around the search.
//   6. Print the timing in the same format as the other benchmark.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>  

// sqlite3.h is the database library; sqlite-vec.h is the vec0's interface
#define SQLITE_CORE
#include "sqlite3.h"
#include "vendor/sqlite-vec.h"

// Helper: if a SQLite call failed, print why and stop the program.
void check(sqlite3* db, int rc, const char* what) {
    if (rc != SQLITE_OK) {
        std::cout << "SQLite error during " << what << ": "
                  << sqlite3_errmsg(db) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // Same command line as bench_cpp.cpp: which vectors file, the dimension,
    // and how many neighbours (k) to return.
    if (argc < 4) {
        std::cout << "usage: " << argv[0]
                  << " <vectors_file> <dim> <k> [metric]\n";
        return 1;
    }
    std::string vectors_path = argv[1];
    int dim = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);
    std::string queries_path = "queries_" + std::to_string(dim) + ".txt";

    // Optional 4th arg picks the distance metric. vec0's default is L2, so
    // Euclidean adds no clause; "cosine" sets distance_metric=cosine on the
    // vector column.
    std::string metric = (argc >= 5) ? argv[4] : "euclidean";
    std::string metric_clause = (metric == "cosine") ? " distance_metric=cosine" : "";

    // --- Step 1: turn on vec0, then open an in-memory database. ---
    sqlite3_auto_extension((void (*)(void))sqlite3_vec_init);

    sqlite3* db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    check(db, rc, "open database");

    // --- Step 2a: create the vec0 table. ---
    std::string create_sql = "CREATE VIRTUAL TABLE vecs USING vec0(embedding float["
                             + std::to_string(dim) + "]" + metric_clause + ")";
    rc = sqlite3_exec(db, create_sql.c_str(), NULL, NULL, NULL);
    check(db, rc, "create table");

    // --- Step 2b: INSERT the stored vectors---
    auto load_start = std::chrono::steady_clock::now();
    sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);

    // Prepare the insert statement once, then reuse it for every row. The two
    // "?" marks are placeholders we fill in per row: the id and the vector.
    sqlite3_stmt* insert = NULL;
    rc = sqlite3_prepare_v2(db,
        "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", -1, &insert, NULL);
    check(db, rc, "prepare insert");

    std::ifstream vfile(vectors_path);
    if (!vfile) {
        std::cout << "Could not open " << vectors_path << "\n";
        return 1;
    }

    
    long long inserted = 0;

    std::string line;
    while (std::getline(vfile, line)) {
        std::istringstream stream(line);

        long long id;
        if (!(stream >> id)) {
            continue;  // skip blank or broken lines
        }

        std::vector<float> values;
        double value;
        while (stream >> value) {
            values.push_back(float(value));
        }

        sqlite3_bind_int64(insert, 1, id);
        sqlite3_bind_blob(insert, 2, values.data(),
                          int(values.size() * sizeof(float)), SQLITE_TRANSIENT);
        rc = sqlite3_step(insert);   // actually perform the insert
        if (rc != SQLITE_DONE) {
            check(db, rc, "insert row");
        }
        sqlite3_reset(insert);       // get the statement ready for the next row
        inserted++;
    }
    sqlite3_finalize(insert);
    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    auto load_end = std::chrono::steady_clock::now();

    // --- Step 3: read the query vectors into memory (as floats). ---
    std::vector<std::vector<float>> queries;
    std::ifstream qfile(queries_path);
    if (!qfile) {
        std::cout << "Could not open " << queries_path << "\n";
        return 1;
    }
    while (std::getline(qfile, line)) {
        std::istringstream stream(line);
        long long id;
        if (!(stream >> id)) {
            continue;
        }
        std::vector<float> values;
        double value;
        while (stream >> value) {
            values.push_back(float(value));
        }
        queries.push_back(values);
    }
    if (queries.empty()) {
        std::cout << "No queries found in " << queries_path << "\n";
        return 1;
    }

    // Prepare the search (KNN = "k nearest neighbours") query once.
    // The two "?" marks are the query vector and the number k.
    sqlite3_stmt* knn = NULL;
    rc = sqlite3_prepare_v2(db,
        "SELECT rowid, distance FROM vecs "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?", -1, &knn, NULL);
    check(db, rc, "prepare knn");

    auto run_one_query = [&](std::size_t i) -> long long {
        long long best_rowid = 0;

        // Fill in the placeholders: the query vector (as raw float bytes) and k.
        sqlite3_bind_blob(knn, 1, queries[i].data(),
                          int(queries[i].size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_int(knn, 2, k);

        // Step through the returned rows. With ORDER By distance, The first row is the closest match.
        bool first = true;
        while (sqlite3_step(knn) == SQLITE_ROW) {
            if (first) {
                best_rowid = sqlite3_column_int64(knn, 0);
                first = false;
            }
        }
        sqlite3_reset(knn);  // ready the statement for the next query
        return best_rowid;
    };

    // --- Step 4: warm-up (not timed), same as the other benchmark. ---
    long long checksum = 0;
    for (std::size_t i = 0; i < 3 && i < queries.size(); ++i) {
        checksum += run_one_query(i);
    }

    // --- Step 5: the timed run. ---
    auto search_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        checksum += run_one_query(i);
    }
    auto search_end = std::chrono::steady_clock::now();

    sqlite3_finalize(knn);
    sqlite3_close(db);

    // --- Step 6: print the results in the SAME format as bench_cpp.cpp. ---
    double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();
    double search_ms =
        std::chrono::duration<double, std::milli>(search_end - search_start)
            .count();
    double per_query_ms = search_ms / queries.size();
    double queries_per_sec = 1000.0 / per_query_ms;

    std::cout << "engine=sqlite_vec"
              << "  N=" << inserted
              << "  dim=" << dim
              << "  k=" << k
              << "  queries=" << queries.size()
              << "  load_ms=" << load_ms
              << "  per_query_ms=" << per_query_ms
              << "  queries_per_sec=" << queries_per_sec
              << "  (checksum=" << checksum << ")\n";
    return 0;
}
