#ifndef DATASET_H
#define DATASET_H

#include <optional>
#include <set>
#include <vector>

#include "bitmask.hpp"
#include "configuration.hpp"
#include "matrix.hpp"

class Dataset {
   public:
    /// Constructs a new dataset (without a reference matrix).
    ///
    /// `feature_map` is provided as original feature -> binarized feature ids to
    /// preserve the Python-facing API, but internally we keep a compact reverse
    /// lookup for binarized -> original feature queries.
    Dataset(const Configuration &config, const Matrix<bool> &input, const Matrix<float> &costs,
            const std::vector<std::set<size_t>> &feature_map);
    /// Constructs a new dataset (with a reference matrix)
    Dataset(const Configuration &config, const Matrix<bool> &input, const Matrix<float> &costs,
            const std::vector<std::set<size_t>> &feature_map, const Matrix<bool> &reference_matrix);

    struct SummaryStatistics {
        float info;
        float potential;
        float max_loss;
        float min_loss;
        float guaranteed_min_loss;
        size_t optimal;
    };
    /// Generates the summarry statistics:
    /// 1. akaike information index
    /// 2. the maximum potential cost reduction between different prediction
    /// choices
    /// 3. equal point loss
    /// 4. maximum loss (classification by the majority target)
    /// 5. the minimum loss incurred by the reference predictions
    /// 6. optimal feature
    SummaryStatistics summary_statistics(const Bitmask &capture_set, Bitmask &work_buffer) const;

    /// Splits the capture_set in place on the given feature_index. If positive
    /// is false, then the subset that is not captured by feature_index is
    /// returned instead.
    void subset_inplace(Bitmask &capture_set, size_t feature_index, bool positive) const;

    float distance(const Bitmask &capture_set, size_t i, size_t j, Bitmask &work_buffer) const;

    [[nodiscard]] size_t original_feature(size_t binarized_feature) const;

    /// Configuration object
    const Configuration &m_config;

    /// Number of rows in the dataset.
    const size_t m_number_rows;

    /// Number of feature columns in the dataset.
    const size_t m_number_features;

    /// Number of target columns in the dataset.
    const size_t m_number_targets;

    /// Save and Load from files.
    void save(const std::string &filename) const;
    static Dataset load(const Configuration &config, const std::string &filename);

   private:
    /// Ctor helpers.
    void initialize_feature_map(const std::vector<std::set<size_t>> &feature_map);
    void construct_bitmasks(const Matrix<bool> &input);
    void construct_cost_matrices(const Matrix<float> &cost_matrix);
    void construct_majority_bitmask();
    void construct_reference_bitmasks(const Matrix<bool> &reference_matrix);

   private:
    /// Row view of the features in a compact binary representation.
    std::vector<Bitmask> m_row_view_features;

    /// Row view of the targets in a compact binary representation.
    std::vector<Bitmask> m_row_view_targets;

    /// Column view of the features in a compact binary representation.
    std::vector<Bitmask> m_col_view_features;

    /// Column view of the targets in a compact binary representation.
    std::vector<Bitmask> m_col_view_targets;

    /// Bitmask marking the rows of the dataset whose target matches the
    /// majority target of the equivalence class of that row (The same
    /// feature row can appear multiple times in a dataset with a different
    /// target value).
    Bitmask m_majority_bitmask;

    /// Cost matrices
    Matrix<float> m_cost_matrix;
    std::vector<float> m_diff_costs;
    std::vector<float> m_match_costs;
    std::vector<float> m_mismatch_costs;

    // Optional Reference Model matrix (Guaranteed to be present if the
    // Configuration flag, `reference_LB` is set).
    std::optional<std::vector<Bitmask>> m_reference_targets;

    // Stable original feature -> binarized feature groups for persistence.
    std::vector<std::vector<size_t>> m_feature_groups;

    // Direct binarized feature -> original feature lookup used at runtime.
    std::vector<size_t> m_binarized_to_original_feature;
};

#endif
