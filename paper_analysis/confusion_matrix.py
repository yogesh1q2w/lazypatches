import numpy as np

def patch_accuracy(mask_method, oracle_T_dimension):
    """
    Computes the patch accuracy

    Parameters:
    mask_method (numpy.ndarray): The method mask of shape (T, H, W).
    mask_oracle (numpy.ndarray): The oracle mask of shape (T).

    Returns:
    float: The ratio of relevant masks in total masks
    """
    assert mask_method.shape[0] == oracle_T_dimension.shape[0], "Masks must have the same shape"
    return np.sum(mask_method[oracle_T_dimension, :, :]) / np.sum(mask_method)

def get_confusion_matrix(is_correct, masks, oracles, is_selected_texts):
    confusion_matrix = {'true_answer': {'true_texts_true_mask': 0,
                                        'true_texts_false_mask': 0,
                                        'false_texts_true_mask': 0,
                                        'false_texts_false_mask': 0},
                        'false_answer': {'true_texts_true_mask': 0,
                                         'true_texts_false_mask': 0,
                                         'false_texts_true_mask': 0,
                                         'false_texts_false_mask': 0}
                        }
    relevant_mask_threshold = 0.3
    for mask, oracle, correct, selected  in zip(masks, oracles, is_correct, is_selected_texts):
        patch_accuracy = patch_accuracy(mask, oracle)
        if correct:
            if selected:
                if patch_accuracy >= relevant_mask_threshold:
                    confusion_matrix['true_answer']['true_texts_true_mask'] += 1
                else:
                    confusion_matrix['true_answer']['true_texts_false_mask'] += 1
            else:
                if patch_accuracy >= relevant_mask_threshold:
                    confusion_matrix['true_answer']['false_texts_true_mask'] += 1
                else:
                    confusion_matrix['true_answer']['false_texts_false_mask'] += 1
        else:
            if selected:
                if patch_accuracy >= relevant_mask_threshold:
                    confusion_matrix['false_answer']['true_texts_true_mask'] += 1
                else:
                    confusion_matrix['false_answer']['true_texts_false_mask'] += 1
            else:
                if patch_accuracy >= relevant_mask_threshold:
                    confusion_matrix['true_answer']['false_texts_true_mask'] += 1
                else:
                    confusion_matrix['true_answer']['false_texts_false_mask'] += 1
    return confusion_matrix