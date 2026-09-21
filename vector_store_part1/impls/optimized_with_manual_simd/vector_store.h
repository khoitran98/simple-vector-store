// vector_store.h
//
// A local vector store. It stores vectors (each with an integer id) and finds
// the closest ones by brute force: it compares your query against EVERY stored
// vector.
//
// This version is tuned for speed. The two ideas behind the speedup are:
//
//   * cache-friendly layout -- every stored vector lives in ONE big flat array
//     (structure-of-arrays), so a search streams straight through contiguous
//     memory instead of chasing a pointer to each row's own allocation.
//   * SIMD -- the inner multiply-add loops use NEON on ARM (for Apple Silicon),
//     doing 4 floats per instruction, with a plain scalar fallback elsewhere.
//   * a bounded priority_queue -- search keeps only the best k results in a size-k heap
//     instead of scoring-then-sorting all N and throwing most away.
//
// Vectors are stored internally as 32-bit floats (like most vector databases),
// but the public API still speaks std::vector<double> similar to the previous implementation.

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <charconv>
#include <queue>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


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

// ----- SIMD kernels (the hot inner loops) ---------------------------------
// These operate on raw float arrays of length `n`. In the store, `n` is always
// the padded stride (a multiple of 4) with the tail zero-filled, so there is no
// leftover "tail" to handle separately: padding a with zeros contributes 0 to
// both the dot product and the squared distance.
//
// We keep FOUR running totals instead of one. Floating-point addition has a few
// cycles of latency, so a single accumulator would stall waiting for each add to
// finish before the next could start. Four independent chains let the CPU keep
// several adds in flight at once, then we combine them at the very end.

#if defined(__ARM_NEON)

// Dot product of two float arrays using NEON (4 floats per register).
inline float dot_simd(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    // Any remaining full groups of 4 (stride is a multiple of 4, tail is zeros).
    for (; i + 4 <= n; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(sum);  // horizontal add of the 4 lanes
}

// Squared Euclidean distance (no sqrt) of two float arrays using NEON.
inline float l2sq_simd(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d, d);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(sum);
}

#else  // portable scalar fallback (still uses 4 accumulators)

inline float dot_simd(const float* a, const float* b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

inline float l2sq_simd(const float* a, const float* b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float d0 = a[i + 0] - b[i + 0];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        s0 += d * d;
    }
    return (s0 + s1) + (s2 + s3);
}

#endif  // __ARM_NEON

// ----- The vector store ---------------------------------------------------

class VectorStore {
public:
    // Create a store where every vector must have exactly `dimension` numbers.
    
    VectorStore(int dimension) {
        if (dimension <= 0) {
            throw std::invalid_argument("dimension must be positive");
        }
        this->dimension = dimension;
        // Round the per-row stride up to a multiple of 4 so the SIMD loop can run
        // whole 4-float groups with no leftover tail. The padding floats are kept
        // at 0, which does not affect dot products or squared distances.
        this->stride = (dimension + 3) / 4 * 4;
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
        append_row(values.data(), int(values.size()));
        ids.push_back(next_id);
        int assigned = next_id;
        next_id = next_id + 1;
        return assigned;
    }

    // Remove the first vector that has this id.
    // Returns true if something was removed, false if the id was not found.
    bool remove(int id) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == id) {
                erase_row(i);
                return true;
            }
        }
        return false;
    }

    
    int size() const {
        return int(ids.size());
    }

    // Read-only listing of the stored records.
    //
    // Unlike the previous implementation, records are no longer kept as a
    // std::vector<Record> internally (they live in one flat float array). This
    // rebuilds that representation on demand, so it costs O(N * dimension). It is
    // only meant for listing/inspection; the search path never calls it.
     std::vector<Record> all() const {
        std::vector<Record> records;
        records.reserve(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            Record record;
            record.id = ids[i];
            record.values = row_to_double(i);
            records.push_back(std::move(record));
        }
        return records;
    }

    // Brute-force search.
    // Compares the query to every stored vector, then returns the best k,
    // with the best match first.
    std::vector<Result> search(const std::vector<double>& query,
                               int k, Metric metric) const {
        std::vector<Result> results;

        
        if (int(query.size()) != dimension) {
            return results;
        }
        if (k <= 0 || ids.empty()) {
            return results;
        }

        // Copy the query into a float buffer once (narrow double->float) so
        // every kernel call reuses it.
        std::vector<float> q(stride, 0.0f);
        for (int j = 0; j < dimension; ++j) {
            q[j] = float(query[j]);
        }

        // For cosine, the query's length is the same for every record, so we
        // compute it a single time.
        float qnorm = 0.0f;
        if (metric == Metric::Cosine) {
            qnorm = std::sqrt(dot_simd(q.data(), q.data(), stride));
        }

        // Score every row into a size-k heap. We store the RANKING score, which
        // for Euclidean is the squared distance (monotonic with real distance)
        // -- the sqrt is applied to the k winners
        // at the very end, not N times inside the loop.
        // For Euclidean, smaller is better; for DotProduct/Cosine, larger is
        // better. A single `better(a, b)` predicate captures the direction.
        const bool smaller_is_better = (metric == Metric::Euclidean);

        // Heap entry: (ranking score, row index).
        struct Cand { float score; int idx; };

        // `better(a, b)` is true when a should rank ahead of b.
         auto better = [smaller_is_better](const Cand& a, const Cand& b) {
            if (smaller_is_better) {
                return a.score < b.score;  
            } else {
                return a.score > b.score;  
            }
        };
        // keeping worst at the top
        std::priority_queue<Cand, std::vector<Cand>, decltype(better)>
            heap(better);

        const int n = int(ids.size());
        for (int i = 0; i < n; ++i) {
            const float* row = data.data() + std::size_t(i) * stride;

            float score;
            if (metric == Metric::Euclidean) {
                score = l2sq_simd(q.data(), row, stride);
            } else if (metric == Metric::DotProduct) {
                score = dot_simd(q.data(), row, stride);
            } else { // Cosine
                float denom = qnorm * norms[i];
                score = (denom == 0.0f)
                            ? 0.0f
                            : dot_simd(q.data(), row, stride) / denom;
            }

            if (int(heap.size()) < k) {
                heap.push(Cand{score, i});
            } else if (better(Cand{score, i}, heap.top())) {
                heap.pop();
                heap.push(Cand{score, i});
            }
        }

        // Drain the heap and build the top k winners
        int found = int(heap.size());
        results.resize(found);
        for (int pos = found - 1; pos >= 0; --pos) {
            const Cand& c = heap.top();
            Result& r = results[pos];
            r.id = ids[c.idx];
            r.values = row_to_double(c.idx);          
            r.score = (metric == Metric::Euclidean)
                          ? std::sqrt(double(c.score))
                          : double(c.score);
            heap.pop();
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
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const float* row = data.data() + i * stride;
            file << ids[i];
            for (int j = 0; j < dimension; ++j) {
                file << " " << double(row[j]);
            }
            file << "\n";
        }
        return true;
    }

    // Load vectors from a text file .
    // This first clears the current store, then reads every line.
    // Returns true on success.
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;  
        }

        // Slurp the whole file into one buffer.
        std::string buf;
        file.seekg(0, std::ios::end);
        std::streamoff len = file.tellg();
        if (len > 0) {
            buf.resize(std::size_t(len));
            file.seekg(0, std::ios::beg);
            file.read(&buf[0], len);
            buf.resize(std::size_t(file.gcount()));
        }

        // Reset the store, then reserve based on a quick newline count so the
        // flat array is allocated once.
        data.clear();
        ids.clear();
        norms.clear();
        next_id = 1;

        std::size_t line_estimate =
            std::size_t(std::count(buf.begin(), buf.end(), '\n')) + 1;
        ids.reserve(line_estimate);
        norms.reserve(line_estimate);
        data.reserve(line_estimate * std::size_t(stride));

        const char* p = buf.data();
        const char* end = p + buf.size();
        std::vector<float> row(stride);

        while (p < end) {
            // Skip leading whitespace / blank lines.
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' ||
                               *p == '\n')) {
                ++p;
            }
            if (p >= end) break;

            // Parse the id (integer).
            int id = 0;
            auto res = std::from_chars(p, end, id);
            if (res.ec != std::errc()) {
                // Not a number -> skip to end of line and continue.
                while (p < end && *p != '\n') ++p;
                continue;
            }
            p = res.ptr;

            // Parse up to `dimension` floats into the row buffer.
            std::fill(row.begin(), row.end(), 0.0f);
            int count = 0;
            while (count < dimension && p < end) {
                // Skip separators, but stop at a newline (end of this record).
                while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
                if (p >= end || *p == '\n') break;

                float value = 0.0f;
                auto fr = std::from_chars(p, end, value);
                if (fr.ec != std::errc()) {
                    // Malformed token -> skip to end of line.
                    while (p < end && *p != '\n') ++p;
                    break;
                }
                p = fr.ptr;
                row[count] = value;
                ++count;
            }

            data.insert(data.end(), row.begin(), row.end());
            ids.push_back(id);
            norms.push_back(std::sqrt(dot_simd(row.data(), row.data(), stride)));

            if (id >= next_id) {
                next_id = id + 1;
            }

            // Advance to the next line.
            while (p < end && *p != '\n') ++p;
        }
        return true;
    }

private:
    // Append one row (given as `count` doubles) to the flat arrays, narrowing to
    // float and caching its L2 norm.
    void append_row(const double* values, int count) {
        std::size_t base = data.size();
        data.resize(base + stride, 0.0f);
        for (int j = 0; j < count && j < stride; ++j) {
            data[base + j] = float(values[j]);
        }
        float* row = data.data() + base;
        norms.push_back(std::sqrt(dot_simd(row, row, stride)));
    }

    void erase_row(std::size_t i) {
        std::size_t base = i * std::size_t(stride);
        data.erase(data.begin() + base, data.begin() + base + stride);
        ids.erase(ids.begin() + i);
        norms.erase(norms.begin() + i);
    }

    // Widen row `i` back to a std::vector<double> of the logical dimension.
    std::vector<double> row_to_double(std::size_t i) const {
        const float* row = data.data() + i * std::size_t(stride);
        std::vector<double> out(dimension);
        for (int j = 0; j < dimension; ++j) {
            out[j] = double(row[j]);
        }
        return out;
    }

    int dimension;             // how many numbers each vector must have
    int stride;                // padded per-row length (multiple of 4)
    int next_id;               // the id the next added vector will get

    // Flat storage. Row i occupies data[i*stride .. i*stride+stride).
    std::vector<float> data;   // all rows back-to-back, tail-padded with zeros
    std::vector<int> ids;      // id for each row
    std::vector<float> norms;  // cached L2 norm of each row (for cosine)
};
