// vector_store.h
//
// A very simple local vector store.
// It stores vectors (each with an integer id) and finds the closest ones
// by brute force: it compares your query against EVERY stored vector.

#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

enum class Metric { Euclidean, DotProduct, Cosine };

// One stored item: an id and its list of numbers.
struct Record {
    int id;
    std::vector<double> values;
};

// One search result: the id of a stored vector, its values, and its score
// vs. the query.
struct Result {
    int id;
    std::vector<double> values;
    double score;
};

// ----- Math helpers -------------------------------------------------

// Euclidean distance: sqrt of the sum of squared differences.
// Smaller means the two vectors are CLOSER.
inline double euclidean_distance(const std::vector<double>& a,
                                 const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Dot product: sum of a[i] * b[i].
// In this context, larger dot product means the two vectors are "nearer".
inline double dot_product(const std::vector<double>& a,
                          const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Length (magnitude) of a vector: sqrt of the dot product with itself.
inline double magnitude(const std::vector<double>& a) {
    return std::sqrt(dot_product(a, a));
}

// Cosine similarity: dot product divided by the two lengths.
// Ranges from -1 to 1; larger (closer to 1) means more similar.
inline double cosine_similarity(const std::vector<double>& a,
                                const std::vector<double>& b) {
    double denom = magnitude(a) * magnitude(b);
    if (denom == 0.0) {
        return 0.0;
    }
    return dot_product(a, b) / denom;
}

// ----- The vector store ---------------------------------------------------

class VectorStore {
public:
    // Create a store where every vector must have exactly `dimension` numbers.
    VectorStore(int dimension) {
        if (dimension <= 0) {
            throw std::invalid_argument("dimension must be positive");
        }
        this->dimension = dimension;
        this->next_id = 1;   // the first vector added will get id 1
    }

    
    int get_dimension() const {
        return dimension;
    }

    // Add a new vector and increment the id
    // Returns the new id, or -1 if the vector has the wrong number of values.
    int add(const std::vector<double>& values) {
        if (int(values.size()) != dimension) {
            return -1;
        }
        Record record;
        record.id = next_id;
        record.values = values;
        records.push_back(record);
        next_id = next_id + 1;
        return record.id;
    }

    // Remove the first vector that has this id.
    // Returns true if something was removed, false if the id was not found.
    bool remove(int id) {
        for (std::size_t i = 0; i < records.size(); ++i) {
            if (records[i].id == id) {
                records.erase(records.begin() + i);
                return true;
            }
        }
        return false;
    }
    
    int size() const {
        return int(records.size());
    }

    const std::vector<Record>& all() const {
        return records;
    }

    // Brute-force search.
    // Compares the query to every stored vector, then returns the best k,
    // with the best match first.
    std::vector<Result> search(const std::vector<double>& query,
                               int k, Metric metric) const {
        std::vector<Result> results;

        
        if (k <= 0 || int(query.size()) != dimension) {
            return results;
        }

        // Step 1: score every stored vector.
        for (std::size_t i = 0; i < records.size(); ++i) {
            const Record& record = records[i];

            double score = 0.0;
            if (metric == Metric::Euclidean) {
                score = euclidean_distance(query, record.values);
            } else if (metric == Metric::DotProduct) {
                score = dot_product(query, record.values);
            } else { // Metric::Cosine
                score = cosine_similarity(query, record.values);
            }

            Result result;
            result.id = record.id;
            result.values = record.values;
            result.score = score;
            results.push_back(result);
        }

        // Step 2: sort so the best match is first.
        // For Euclidean, smaller distance is better (ascending).
        // For DotProduct and Cosine, larger score is better (descending).
        if (metric == Metric::Euclidean) {
            std::sort(results.begin(), results.end(),
                      [](const Result& a, const Result& b) {
                          return a.score < b.score;
                      });
        } else {
            std::sort(results.begin(), results.end(),
                      [](const Result& a, const Result& b) {
                          return a.score > b.score;
                      });
        }

        // Step 3: keep only the first k results.
        if (int(results.size()) > k) {
            results.resize(k);
        }
        return results;
    }

    // Save all vectors to a text file.
    // Each line looks like:  id  v0  v1  v2 ...
    // Returns true on success.
    bool save(const std::string& path) const {
        std::ofstream file(path);
        if (!file) {
            return false;  
        }
        for (std::size_t i = 0; i < records.size(); ++i) {
            const Record& record = records[i];
            file << record.id;
            for (std::size_t j = 0; j < record.values.size(); ++j) {
                file << " " << record.values[j];
            }
            file << "\n";
        }
        return true;
    }

    // Load vectors from a text file written by save().
    // This first clears the current store, then reads every line.
    // Returns true on success.
    bool load(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            return false;  
        }

        records.clear();
        next_id = 1;

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line);

            int id;
            if (!(stream >> id)) {
                continue;  // skip empty or bad lines
            }

            std::vector<double> values;
            double value;
            while (stream >> value) {
                values.push_back(value);
            }

            Record record;
            record.id = id;
            record.values = values;
            records.push_back(record);

            if (id >= next_id) {
                next_id = id + 1;
            }
        }
        return true;
    }

private:
    int dimension;                    // how many numbers each vector must have
    int next_id;                      // the id the next added vector will get
    std::vector<Record> records;
};

#endif  // VECTOR_STORE_H
