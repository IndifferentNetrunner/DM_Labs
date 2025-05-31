import time
from itertools import combinations
from collections import defaultdict, Counter

class Apriori:
    def __init__(self, min_support=0.3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.freq_itemsets = []
        self.rules = []

    def fit(self, transactions):
        self.transactions = list(map(set, transactions))
        self.num_transactions = len(self.transactions)
        self.freq_itemsets = []
        self.rules = []
        self._apriori()

    def _apriori(self):
        print("[Apriori] Counting individual items...")
        item_counts = Counter(item for transaction in self.transactions for item in transaction)
        current_itemsets = [{item} for item, count in item_counts.items() if count / self.num_transactions >= self.min_support]
        print(f"[Apriori] Found {len(current_itemsets)} frequent 1-itemsets: {current_itemsets}")
        self.freq_itemsets.extend(current_itemsets)

        k = 2
        while current_itemsets:
            print(f"[Apriori] Generating candidates of size {k}...")
            candidates = self._generate_candidates(current_itemsets, k)
            print(f"[Apriori] {len(candidates)} candidates generated: {candidates}")
            itemset_counts = Counter()
            for transaction in self.transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        itemset_counts[frozenset(candidate)] += 1
            current_itemsets = [set(itemset) for itemset, count in itemset_counts.items() if count / self.num_transactions >= self.min_support]
            print(f"[Apriori] {len(current_itemsets)} frequent itemsets of size {k} found: {current_itemsets}")
            self.freq_itemsets.extend(current_itemsets)
            k += 1

        print("[Apriori] Generating association rules...")
        self._generate_rules()
        print(f"[Apriori] {len(self.rules)} rules generated")

    def _generate_candidates(self, itemsets, k):
        candidates = []
        len_itemsets = len(itemsets)
        for i in range(len_itemsets):
            for j in range(i + 1, len_itemsets):
                union_set = itemsets[i] | itemsets[j]
                if len(union_set) == k and union_set not in candidates:
                    candidates.append(union_set)
        return candidates

    def _generate_rules(self):
        self.rules = []
        for itemset in self.freq_itemsets:
            if len(itemset) < 2:
                continue
            itemset_support = sum(1 for t in self.transactions if itemset.issubset(t)) / self.num_transactions
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = set(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = sum(1 for t in self.transactions if antecedent.issubset(t)) / self.num_transactions
                    if antecedent_support > 0:
                        confidence = itemset_support / antecedent_support
                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, itemset_support, confidence))

class FPTreeNode:
    def __init__(self, item, parent):
        self.item = item
        self.parent = parent
        self.count = 1
        self.children = {}
        self.link = None

class FPGrowth:
    def __init__(self, min_support=0.3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.freq_itemsets = []
        self.rules = []

    def fit(self, transactions):
        self.transactions = list(map(set, transactions))
        self.num_transactions = len(self.transactions)
        self.freq_itemsets = []
        self.rules = []

        print("[FP-Growth] Counting item frequencies...")
        item_counts = Counter(item for transaction in self.transactions for item in transaction)
        self.freq_items = {item for item, count in item_counts.items() if count / self.num_transactions >= self.min_support}

        print(f"[FP-Growth] Frequent items (support >= {self.min_support*100:.0f}%): {self.freq_items}")

        self.headers = defaultdict(list)

        def sort_items(transaction):
            return sorted([item for item in transaction if item in self.freq_items], key=lambda i: -item_counts[i])

        self.tree = FPTreeNode(None, None)
        for transaction in self.transactions:
            sorted_items = sort_items(transaction)
            self._insert_tree(sorted_items, self.tree)

        print("[FP-Growth] Mining FP-tree...")
        self._mine_tree(self.headers, set(), item_counts)

        print("[FP-Growth] Generating association rules...")
        self._generate_rules()
        print(f"[FP-Growth] {len(self.rules)} rules generated")

    def _insert_tree(self, items, node):
        if not items:
            return
        first, rest = items[0], items[1:]
        if first in node.children:
            node.children[first].count += 1
        else:
            node.children[first] = FPTreeNode(first, node)
            self.headers[first].append(node.children[first])
        self._insert_tree(rest, node.children[first])

    def _mine_tree(self, headers, prefix, item_counts):
        # Відключаємо рекурсивний виклик fit, переписуємо на нормальне добування
        sorted_items = sorted(headers.keys(), key=lambda i: item_counts[i])
        for item in sorted_items:
            new_prefix = prefix | {item}
            support = sum(node.count for node in headers[item]) / self.num_transactions
            if support >= self.min_support:
                self.freq_itemsets.append(new_prefix)

                # Збираємо conditional patterns
                conditional_patterns = []
                for node in headers[item]:
                    path = []
                    parent = node.parent
                    while parent and parent.item is not None:
                        path.append(parent.item)
                        parent = parent.parent
                    path.reverse()
                    for _ in range(node.count):
                        conditional_patterns.append(path)

                # Побудова conditional FP-tree
                if conditional_patterns:
                    conditional_tree = FPGrowth(self.min_support, self.min_confidence)
                    conditional_tree.fit(conditional_patterns)
                    for itemset in conditional_tree.freq_itemsets:
                        self.freq_itemsets.append(new_prefix | itemset)

    def _generate_rules(self):
        self.rules = []
        for itemset in self.freq_itemsets:
            if len(itemset) < 2:
                continue
            itemset_support = sum(1 for t in self.transactions if itemset.issubset(t)) / self.num_transactions
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = set(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = sum(1 for t in self.transactions if antecedent.issubset(t)) / self.num_transactions
                    if antecedent_support > 0:
                        confidence = itemset_support / antecedent_support
                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, itemset_support, confidence))

def run_comparison():
    transactions = [
        {'milk', 'bread', 'eggs', 'apple'},
        {'milk', 'bread', 'butter'},
        {'milk', 'bread', 'apple', 'butter'},
        {'milk', 'eggs', 'butter', 'orange'},
        {'bread', 'apple', 'eggs', 'banana'},
        {'milk', 'banana', 'butter', 'orange'},
        {'bread', 'butter', 'orange', 'apple'},
        {'eggs', 'apple', 'orange', 'banana'},
        {'milk', 'bread', 'butter', 'banana'},
        {'bread', 'apple', 'butter', 'orange'},
        {'milk', 'apple', 'banana', 'butter'},
        {'milk', 'bread', 'apple', 'eggs', 'butter'},
        {'bread', 'eggs', 'banana', 'orange'},
        {'milk', 'eggs', 'apple', 'banana'},
        {'milk', 'bread', 'apple', 'eggs', 'butter', 'banana'},
        {'bread', 'apple', 'eggs', 'orange'},
        {'milk', 'apple', 'orange', 'banana'},
        {'bread', 'butter', 'eggs', 'apple'},
        {'milk', 'bread', 'eggs', 'banana'},
        {'apple', 'butter', 'orange', 'banana'},
        {'milk', 'bread', 'butter', 'eggs', 'apple'},
        {'milk', 'banana', 'eggs', 'orange'},
        {'bread', 'apple', 'butter', 'banana'},
        {'milk', 'eggs', 'apple', 'orange'},
        {'bread', 'apple', 'butter', 'banana', 'orange'},
        {'milk', 'bread', 'eggs', 'butter', 'apple'},
        {'apple', 'banana', 'orange', 'butter'},
        {'milk', 'bread', 'apple', 'banana', 'eggs'},
        {'milk', 'orange', 'banana', 'butter'},
        {'bread', 'apple', 'eggs', 'butter'},
        {'milk', 'bread', 'apple', 'eggs', 'banana'},
        {'bread', 'butter', 'orange', 'apple'}
    ]

    print("\n=== Running Apriori ===")
    apriori = Apriori(min_support=0.3, min_confidence=0.7)
    start = time.time()
    apriori.fit(transactions)
    end = time.time()
    print(f"Frequent itemsets (support >= 30%):")
    for itemset in apriori.freq_itemsets:
        support = sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)
        print(f"{itemset}: {support:.2%}")
    print(f"Apriori found {len(apriori.rules)} rules in {end-start:.4f} seconds\n")

    print("\n=== Running FP-Growth ===")
    fp_growth = FPGrowth(min_support=0.3, min_confidence=0.7)
    start = time.time()
    fp_growth.fit(transactions)
    end = time.time()
    print(f"Frequent itemsets (support >= 30%):")
    for itemset in fp_growth.freq_itemsets:
        support = sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)
        print(f"{itemset}: {support:.2%}")
    print(f"FP-Growth found {len(fp_growth.rules)} rules in {end-start:.4f} seconds\n")

if __name__ == "__main__":
    run_comparison()