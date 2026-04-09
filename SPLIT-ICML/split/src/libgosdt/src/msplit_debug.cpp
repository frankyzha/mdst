    static nlohmann::json to_json(const std::shared_ptr<Node> &node) {
        if (!node) {
            return nlohmann::json::object();
        }
        if (node->is_leaf) {
            nlohmann::json class_counts = node->class_counts.empty()
                ? nlohmann::json::array({node->neg_count, node->pos_count})
                : nlohmann::json(node->class_counts);
            return nlohmann::json{
                {"type", "leaf"},
                {"prediction", node->prediction},
                {"loss", node->loss},
                {"n_samples", node->n_samples},
                {"class_counts", std::move(class_counts)},
            };
        }

        nlohmann::json groups = nlohmann::json::array();
        for (size_t i = 0; i < node->group_nodes.size(); ++i) {
            nlohmann::json spans = nlohmann::json::array();
            for (const auto &span : node->group_bin_spans[(size_t)i]) {
                spans.push_back(nlohmann::json::array({span.first, span.second}));
            }
            groups.push_back(nlohmann::json{
                {"spans", std::move(spans)},
                {"child", to_json(node->group_nodes[(size_t)i])},
            });
        }

        return nlohmann::json{
            {"type", "node"},
            {"feature", node->feature},
            {"fallback_bin", node->fallback_bin},
            {"fallback_prediction", node->fallback_prediction},
            {"group_count", node->group_count},
            {"n_samples", node->n_samples},
            {"groups", std::move(groups)},
        };
    }
