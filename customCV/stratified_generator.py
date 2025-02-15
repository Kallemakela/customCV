import numpy as np
import random
from itertools import combinations
from collections import Counter, defaultdict


class StratifiedCombinationGenerator:
    """
    Generates stratified combinations from an iterable, ensuring a specified
    class distribution within each combination.

    Args:
        iterable: A list of tuples, where each tuple is (sample_index, class_label).
        r: The size of each combination (fold size).
        class_counts: Dictionary of {class_label: total_count}.
        overall_ratios: Dictionary of {class_label: overall_ratio}.
        rng: The random number generator.
    """

    def __init__(self, iterable, r, class_counts, overall_ratios, rng=None):
        self.iterable = iterable
        self.r = r
        self.rng = rng
        self.class_counts = class_counts
        self.overall_ratios = overall_ratios
        self.samples_by_class = self._group_samples_by_class()
        self.target_counts = self._calculate_target_counts()
        self.sorted_classes = sorted(class_counts.keys(), key=lambda k: class_counts[k])

    def _group_samples_by_class(self):
        """Groups samples by their class label."""
        samples_by_class = defaultdict(list)
        for sample, label in self.iterable:
            samples_by_class[label].append(sample)
        return samples_by_class

    def _calculate_target_counts(self):
        """Calculates the target number of samples from each class."""
        target_counts = {
            label: max(0, round(self.overall_ratios[label] * self.r))
            for label in self.overall_ratios
        }

        # Handle rounding errors
        total_target = sum(target_counts.values())
        if total_target != self.r:
            diff = self.r - total_target
            remainders = {
                label: (self.overall_ratios[label] * self.r)
                - round(self.overall_ratios[label] * self.r)
                for label in self.overall_ratios
            }
            sorted_remainders = sorted(
                remainders.items(), key=lambda item: item[1], reverse=True
            )
            for i in range(abs(diff)):
                label = sorted_remainders[i % len(sorted_remainders)][0]
                target_counts[label] += 1 if diff > 0 else -1
        return target_counts

    def _shuffle_samples(self):
        """Shuffles the samples within each class."""
        for label in self.samples_by_class:
            self.rng.shuffle(self.samples_by_class[label])

    def _generate_recursive(self, current_combination, remaining_classes):
        """Recursively generates stratified combinations."""
        if not remaining_classes:
            if len(current_combination) == self.r:
                yield tuple(sorted(current_combination))
            return

        class_label = remaining_classes[0]
        available_samples = self.samples_by_class[
            class_label
        ]  # Use the shuffled list directly
        max_to_take = min(
            len(available_samples),
            self.target_counts[class_label]
            - Counter(x[1] for x in current_combination)[class_label],
        )

        for num_to_take in range(max_to_take + 1):
            for class_comb in combinations(available_samples, num_to_take):
                new_combination = current_combination + [
                    (sample, class_label) for sample in class_comb
                ]
                yield from self._generate_recursive(
                    new_combination, remaining_classes[1:]
                )

    def generate(self):
        """Generates stratified combinations."""
        if self.rng is not None:
            self._shuffle_samples()
        yield from self._generate_recursive([], self.sorted_classes)


def stratified_combinations(iterable, r, rng=None):
    """
    Wrapper function for StratifiedCombinationGenerator.
    """
    class_counts = Counter(y for _, y in iterable)
    overall_ratios = {
        label: count / len(iterable) for label, count in class_counts.items()
    }
    generator = StratifiedCombinationGenerator(
        iterable, r, class_counts, overall_ratios, rng
    )
    return generator.generate()


# rng = np.random.default_rng(0)
# n_samples = 40
# n_splits = 10
# n_repeats = 5
# n_classes = 4
# y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
# u, c = np.unique(y, return_counts=True)
# print(u, c)
# X = np.arange(n_samples)
# class_counts = dict(Counter(y))
# overall_ratios = {label: count / n_samples for label, count in class_counts.items()}
# iterable = [(i, y[i]) for i in range(n_samples)]
# g = stratified_combinations(
#     iterable, n_samples // n_splits, rng, class_counts, overall_ratios
# )

# for i, fold in enumerate(g):
#     print(f"Fold {i}: {fold}")
#     if i > 10:
#         break
