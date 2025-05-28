import math
import random
from collections import Counter
from sklearn.datasets import make_classification

# --- One Rule ---
def one_rule(data, num_attributes):
    rules_list = []
    errors_list = []

    for attr_idx in range(num_attributes):
        rule = {}
        error = 0
        freq = {}

        for row in data:
            val, label = row[attr_idx], row[-1]
            freq.setdefault(val, Counter())[label] += 1

        for val, label_counts in freq.items():
            majority = label_counts.most_common(1)[0][0]
            rule[val] = majority
            error += sum(label_counts.values()) - label_counts[majority]

        rules_list.append(rule)
        errors_list.append(error)

    best_attr = errors_list.index(min(errors_list))
    return best_attr, rules_list[best_attr]

def one_rule_batch_classifier(instances, data):
    attr_idx, rule = one_rule(data, len(data[0]) - 1)
    predictions = []
    for instance in instances:
        predictions.append(rule.get(instance[attr_idx], 0))
    return attr_idx, rule, predictions

# --- Naive Bayes ---
def naive_bayes_classifier(train_data, new_instance, laplace=1e-5):
    n_attrs = len(train_data[0]) - 1
    classes = [row[-1] for row in train_data]
    class_counts = Counter(classes)
    n_total = len(train_data)

    priors = {c: count / n_total for c, count in class_counts.items()}
    cond_probs = {}

    for attr in range(n_attrs):
        cond_probs[attr] = {}
        for c in class_counts:
            class_subset = [row for row in train_data if row[-1] == c]
            values = [row[attr] for row in class_subset]
            val_counts = Counter(values)
            total = len(class_subset)
            cond_probs[attr][c] = {
                val: (val_counts[val] + laplace) / (total + laplace * len(set(values)))
                for val in set(values)
            }

    scores = {}
    for c in class_counts:
        prob = priors[c]
        for attr, val in enumerate(new_instance):
            prob *= cond_probs[attr][c].get(val, laplace)
        scores[c] = prob

    return max(scores, key=scores.get), scores

def naive_bayes_batch_classifier(train_data, instances):
    preds = []
    probs = []
    for instance in instances:
        pred, score = naive_bayes_classifier(train_data, instance)
        preds.append(pred)
        probs.append(score)
    return preds, probs

# --- Decision Tree (ID3) ---
class TreeNode:
    def __init__(self, is_leaf=False, label=None, feature=None):
        self.is_leaf = is_leaf
        self.label = label
        self.feature = feature
        self.children = {}

def entropy(counts, total):
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)

def info_gain(data, attr, target_idx):
    total = len(data)
    parent_classes = Counter(row[target_idx] for row in data)
    base_entropy = entropy(parent_classes, total)

    splits = {}
    for row in data:
        key = row[attr]
        splits.setdefault(key, []).append(row)

    weighted = sum(
        (len(subset) / total) * entropy(Counter(row[target_idx] for row in subset), len(subset))
        for subset in splits.values()
    )
    return base_entropy - weighted

def majority_class(data, target_idx):
    return Counter(row[target_idx] for row in data).most_common(1)[0][0]

def build_decision_tree(data, features, target_idx):
    labels = [row[target_idx] for row in data]
    if len(set(labels)) == 1:
        return TreeNode(is_leaf=True, label=labels[0])
    if not features:
        return TreeNode(is_leaf=True, label=majority_class(data, target_idx))

    gains = [(attr, info_gain(data, attr, target_idx)) for attr in features]
    best_attr = max(gains, key=lambda x: x[1])[0]
    node = TreeNode(feature=best_attr)

    values = set(row[best_attr] for row in data)
    for val in values:
        subset = [row for row in data if row[best_attr] == val]
        if not subset:
            node.children[val] = TreeNode(is_leaf=True, label=majority_class(data, target_idx))
        else:
            sub_features = [f for f in features if f != best_attr]
            node.children[val] = build_decision_tree(subset, sub_features, target_idx)

    return node

def classify_tree(instance, tree):
    if tree.is_leaf:
        return tree.label
    val = instance[tree.feature]
    if val in tree.children:
        return classify_tree(instance, tree.children[val])
    else:
        return 0

def decision_tree_batch_classifier(instances, tree):
    return [classify_tree(instance, tree) for instance in instances]

# --- kNN ---
def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_classify(train_data, new_instance, k=3):
    distances = [(euclidean(row[:-1], new_instance), row[-1]) for row in train_data]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    vote_counts = Counter(label for _, label in k_nearest)
    return vote_counts.most_common(1)[0][0], k_nearest

def knn_batch_classifier(train_data, instances, k=3):
    preds = []
    neighbors_info = []
    for instance in instances:
        pred, neighbors = knn_classify(train_data, instance, k)
        preds.append(pred)
        neighbors_info.append(neighbors)
    return preds, neighbors_info

# --- Data Generation ---
def generate_data(samples=50, features=4):
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],  # Збалансовані класи
        random_state=42
    )
    X = [[int(round(x)) for x in sample] for sample in X]
    return [tuple(x + [int(label)]) for x, label in zip(X, y)]

# --- Metrics ---
def accuracy_score(true_labels, predicted_labels):
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    return correct / len(true_labels)

def precision_recall_f1(true_labels, predicted_labels, positive_class=1):
    tp = sum((t == p == positive_class) for t, p in zip(true_labels, predicted_labels))
    fp = sum((t != positive_class and p == positive_class) for t, p in zip(true_labels, predicted_labels))
    fn = sum((t == positive_class and p != positive_class) for t, p in zip(true_labels, predicted_labels))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# --- Main Execution ---
if __name__ == "__main__":
    # Великий датасет (500 зразків, 4 фічі)
    dataset = generate_data(500, 4)

    # Розділяємо на train/test (80/20)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    test_features = [row[:-1] for row in test_data]
    test_labels = [row[-1] for row in test_data]

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}\n")

    # One Rule
    attr_idx, rule, one_rule_preds = one_rule_batch_classifier(test_features, train_data)
    one_p, one_r, one_f1 = precision_recall_f1(test_labels, one_rule_preds)

    # Naive Bayes
    nb_preds, nb_probs = naive_bayes_batch_classifier(train_data, test_features)
    nb_p, nb_r, nb_f1 = precision_recall_f1(test_labels, nb_preds)

    # Decision Tree
    feature_indices = list(range(len(test_features[0])))
    decision_tree = build_decision_tree(train_data, feature_indices, len(test_features[0]))
    dt_preds = decision_tree_batch_classifier(test_features, decision_tree)
    dt_p, dt_r, dt_f1 = precision_recall_f1(test_labels, dt_preds)

    # kNN
    knn_preds, knn_neighbors = knn_batch_classifier(train_data, test_features, k=5)
    knn_p, knn_r, knn_f1 = precision_recall_f1(test_labels, knn_preds)

    # Вивід результатів
    print("--- One Rule ---")
    print(f"Best attribute index: {attr_idx}, Rule: {rule}")
    print(f"Precision: {one_p:.3f}, Recall: {one_r:.3f}, F1: {one_f1:.3f}\n")

    print("--- Naive Bayes ---")
    print(f"Precision: {nb_p:.3f}, Recall: {nb_r:.3f}, F1: {nb_f1:.3f}\n")

    print("--- Decision Tree ---")
    print(f"Precision: {dt_p:.3f}, Recall: {dt_r:.3f}, F1: {dt_f1:.3f}\n")

    print("--- kNN (k=5) ---")
    print(f"Precision: {knn_p:.3f}, Recall: {knn_r:.3f}, F1: {knn_f1:.3f}\n")

    # Підсумкова таблиця
    print("=== Summary ===")
    print(f"{'Classifier':<15}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-score':<10}")
    print(f"{'One Rule':<15}{accuracy_score(test_labels, one_rule_preds):<10.3f}"
          f"{one_p:<10.3f}{one_r:<10.3f}{one_f1:<10.3f}")
    print(f"{'Naive Bayes':<15}{accuracy_score(test_labels, nb_preds):<10.3f}"
          f"{nb_p:<10.3f}{nb_r:<10.3f}{nb_f1:<10.3f}")
    print(f"{'Decision Tree':<15}{accuracy_score(test_labels, dt_preds):<10.3f}"
          f"{dt_p:<10.3f}{dt_r:<10.3f}{dt_f1:<10.3f}")
    print(f"{'kNN':<15}{accuracy_score(test_labels, knn_preds):<10.3f}"
          f"{knn_p:<10.3f}{knn_r:<10.3f}{knn_f1:<10.3f}")