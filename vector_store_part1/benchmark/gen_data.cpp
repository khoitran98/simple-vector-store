// It writes two plain-text files, both in the same "id v0 v1 v2 ..." format
// that our vector store already understands:
//
//   vectors_<N>_<dim>.txt   -> the N vectors we will store and search through
//   queries_<dim>.txt       -> a set of query vectors we will search FOR
//
//
// Both the C++ benchmark and the sqlite-vec benchmark read these
// SAME files.
//
// The numbers are random, BUT they come from a FIXED starting seed. Same 
// seed => the exact same "random" numbers every time. So re-running 
// this program always rebuilds
// the same files, which makes the benchmark repeatable.

#include <iostream>   
#include <fstream>    
#include <random>     
#include <string>    

// How many query vectors to generate.
const int QUERY_COUNT = 100;

// Write `count` vectors, each with `dim` numbers, into the file at `path`.
// The ids are simply 1, 2, 3, ... up to `count`.
void write_file(const std::string& path, int count, int dim,
                std::mt19937& engine) {
    // This describes the shape of the random numbers: any real number
    // in [-1.0, 1.0), each equally likely. (real embeddings are fairly different,
    // but it works for just evaluating brute-force search runtime)
    std::uniform_real_distribution<double> pick(-1.0, 1.0);

    std::ofstream file(path);

    file.precision(9);

    // One line per vector: first the id, then its `dim` numbers.
    for (int id = 1; id <= count; ++id) {
        file << id;                       // the vector's id
        for (int j = 0; j < dim; ++j) {
            file << " " << pick(engine);  // one random value
        }
        file << "\n";                     
    }
}

int main(int argc, char** argv) {
    // We expect two pieces of information on the command line:
    //   argv[1] = N   (how many vectors to store)
    //   argv[2] = dim (how many numbers each vector has)
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " <N> <dim>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);

    // The random-number machine, started from a FIXED seed (12345).
    std::mt19937 engine(12345);

    // The queries get their OWN generator, seeded independently.
    std::mt19937 query_engine(67890);

    // Build the two file names, e.g. "vectors_10000_128.txt" and
    // "queries_128.txt". Putting N and dim in the name keeps vector store
    // datasets of different sizes from overwriting each other.
    std::string vectors_path =
        "vectors_" + std::to_string(n) + "_" + std::to_string(dim) + ".txt";
    std::string queries_path = "queries_" + std::to_string(dim) + ".txt";

    // Actually create the two files. Each draws from its own generator, so the
    // vectors depend on (n, dim) and the queries depend on dim alone
    write_file(vectors_path, n, dim, engine);
    write_file(queries_path, QUERY_COUNT, dim, query_engine);

    std::cout << "Wrote " << vectors_path << " (" << n << " vectors) and "
              << queries_path << " (" << QUERY_COUNT << " queries), dim "
              << dim << ".\n";
    return 0;
}
