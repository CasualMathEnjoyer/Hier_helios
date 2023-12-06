import numpy as np
def calculate_precision_recall_f1(y_true, y_pred, label):
    true_positive = np.sum((y_true == label) & (y_pred == label))
    false_positive = np.sum((y_true != label) & (y_pred == label))
    false_negative = np.sum((y_true == label) & (y_pred != label))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def multiclass_f1_score(y_true, y_pred):
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))

    total_precision = 0
    total_recall = 0

    for label in unique_labels:
        precision, recall, _ = calculate_precision_recall_f1(y_true, y_pred, label)
        total_precision += precision
        total_recall += recall

    macro_precision = total_precision / len(unique_labels) if len(unique_labels) > 0 else 0
    macro_recall = total_recall / len(unique_labels) if len(unique_labels) > 0 else 0

    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    return macro_f1

# Example usage:
# Assume y_true and y_pred are your true and predicted labels, respectively.

# Generating example labels for demonstration
np.random.seed(2345322)
y_true = np.random.randint(0, 3, size=100)  # 3 classes
y_pred = np.random.randint(0, 3, size=100)

# Calculate multiclass F1 score
f1 = multiclass_f1_score(y_true, y_pred)

print(f'Multiclass F1 Score: {f1}')