import tensorflow as tf
from rouge_score import rouge_scorer

def rem_zero(list):
    list_new = []
    for i in range(len(list)):

        if list[i] == 0:
            break
        list_new.append(list[i])
    return list_new


def test_accuracy(y_true, y_pred):

    y_true = rem_zero(y_true)
    y_pred = rem_zero(y_pred)
    # bow_y_true = set(y_true)
    # bow_y_pred = set(y_pred)

    if len(y_true) < len(y_pred):
        for i in range(len(y_pred) - len(y_true)):
            y_true.append(0)
    if len(y_pred) < len(y_true):
        for i in range(len(y_true) - len(y_pred)):
            y_pred.append(0)

    metric = tf.keras.metrics.Accuracy()
    metric.update_state(y_true, y_pred)

    return metric.result().numpy()

def rouge_evauation(y_true, y_pred, tokenizer):
    rouge1_precision = []
    rouge1_recall = []
    rouge1_f = []
    rouge2_precision = []
    rouge2_recall = []
    rouge2_f = []
    rougeL_precision = []
    rougeL_recall = []
    rougeL_f = []
    for i,j in zip(y_true, y_pred):

        y_t = tokenizer.decode(rem_zero(i))
        y_p = tokenizer.decode(rem_zero(j))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(y_t, y_p)


        rouge1_precision.append(scores['rouge1'].precision)
        rouge1_recall.append(scores['rouge1'].recall)
        rouge1_f.append(scores['rouge1'].fmeasure)

        rouge2_precision.append(scores['rouge2'].precision)
        rouge2_recall.append(scores['rouge2'].recall)
        rouge2_f.append(scores['rouge2'].fmeasure)

        rougeL_precision.append(scores['rougeL'].precision)
        rougeL_recall.append(scores['rougeL'].recall)
        rougeL_f.append(scores['rougeL'].fmeasure)

    return scores, rouge1_precision, rouge1_recall, rouge1_f, rouge2_precision, rouge2_recall, rouge2_f, rougeL_precision, rougeL_recall, rougeL_f



def sparse_accuracy(y_true, y_pred):
    metric = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    return metric





