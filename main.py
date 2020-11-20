import argparse
import tensorflow as tf
import numpy as np
import datetime
from rouge_score import rouge_scorer

tf.random.set_seed(1234)

from model import transformer
from dataset import get_dataset, preprocess_sentence

import testing

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, hparams, warmup_steps=4000):

    super(CustomSchedule, self).__init__()

    self.d_model = tf.cast(hparams.d_model, dtype=tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):

    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def inference(hparams, model, tokenizer, sentence):

  sentence = preprocess_sentence(sentence)


  sentence = tf.expand_dims(
      hparams.start_token + tokenizer.encode(sentence) + hparams.end_token,
      axis=0)


  output = tf.expand_dims(hparams.start_token, 0)

  for i in range(hparams.max_length):

    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, hparams.end_token[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(hparams, model, tokenizer, sentence):

  prediction = inference(hparams, model, tokenizer, sentence)
  # print("predict function ", prediction )

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])


  return predicted_sentence



def evaluate(hparams, model, tokenizer):
  print('\nEvaluate')
  sentence = 'i guess i thought i was protecting you .'
  output = predict(hparams, model, tokenizer, sentence)
  print('input: {}\noutput: {}'.format(sentence, output))

  sentence = "it's a trap!"
  output = predict(hparams, model, tokenizer, sentence)
  print('\ninput: {}\noutput: {}'.format(sentence, output))

  sentence = 'I am not crazy, my mother had me tested'
  for _ in range(5):
    output = predict(hparams, model, tokenizer, sentence)
    print('\ninput: {}\noutput: {}'.format(sentence, output))
    sentence = output


def main(hparams):
    print("Layers", hparams.num_layers)
    train_dataset, test_dataset, tokenizer = get_dataset(hparams)
    dataset = train_dataset

    model = transformer(hparams)

    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(hparams), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def loss_function(y_true, y_pred):
        # print(y_true,"\n" ,y_pred)
        y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
        # print(y_true,"\n")
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))

        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer, loss=loss_function, metrics=[accuracy])

    model.fit(dataset, epochs=hparams.epochs)

    evaluate(hparams, model, tokenizer)

"""
    ## Test Dataset Evaluation

    def test_inference(hparams, sent):
        output = tf.expand_dims(hparams.start_token, 0)
        prob = tf.reshape(tf.cast(np.zeros(hparams.vocab_size), dtype=tf.float32), (1, hparams.vocab_size))
        # print(output)
        for i in range(hparams.max_length):
            predictions = model(inputs=[sent, output], training=False)
            # print(predictions)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, hparams.end_token[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)


        ## for sparse categorial
        output2 = tf.expand_dims(hparams.start_token, 0)
        for i in range(hparams.max_length):
            predictions = model(inputs=[sent, output2], training=False)
            # print(predictions)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, hparams.end_token[0]):
                break
            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output2 = tf.concat([output2, predicted_id], axis=-1)
            pred = tf.squeeze(predictions, axis=0)

            prob = tf.concat([prob, pred], axis=0)

        return output, prob


    i = list(test_dataset.as_numpy_iterator())[0][0]

    j = list(test_dataset.as_numpy_iterator())[0][1]

    # print(i,j)
    ## Getting y_pred for input given by i['input']
    y_pred = tf.reshape(tf.cast(np.zeros(41), dtype=tf.int32), (1, 41))

    for ii in j:
        for k in range(len(ii)):
            if ii[k] == hparams.end_token:
                ii[k] = 0

    y_true = j

    sparse_accu = []

    for sent, true in zip(i['input'], y_true):
        # sent = tf.reshape(sent, [1, len(sent)])
        sent = tf.expand_dims(sent, axis=0)
        true = testing.rem_zero(true)
        pred, prob = test_inference(hparams, sent)

        # print(type(y_pred), type(pred))
        pred = tf.keras.preprocessing.sequence.pad_sequences(pred, maxlen=hparams.max_length + 1, padding='post')
        y_pred = tf.concat([y_pred, pred], axis=0)

        #for sparse
        prob = prob[1:]

        if len(true) < len(prob):
            for i in range(len(prob) - len(true)):
                true.append(0)

        if len(prob) < len(true):
            for i in range(len(true) - len(prob)):
                prob_init = tf.reshape(tf.cast(np.zeros(hparams.vocab_size), dtype=tf.float32), (1, hparams.vocab_size))
                prob = tf.concat([prob, prob_init], axis=0)

        true = tf.cast(true, dtype=tf.float32)

        # print(true, prob)

        m = tf.keras.metrics.sparse_categorical_accuracy(true, prob)
        m = m.numpy()
        # print(m)
        accu = (m == 1).sum() / len(m)
        sparse_accu.append(accu)


    #
    y_pred = y_pred[1:, 1:-1]
    y_pred = y_pred.numpy()



    ## Simple Accuracy and Sparse Categorial accuracy

    accuracy_vector = []

    for i, j in zip(y_true, y_pred):
        accu = testing.test_accuracy(i, j)
        accuracy_vector.append(accu)



    ## Rouge Score
    scores, rouge1_precision, rouge1_recall, rouge1_f, rouge2_precision, rouge2_recall, rouge2_f,rougeL_precision, rougeL_recall, rougeL_f = testing.rouge_evauation(y_true, y_pred, tokenizer)

    print("accuracy = ", accuracy_vector, '\n',
          "rouge1_precision = ",rouge1_precision, '\n',
          "rouge1_recall = ", rouge1_recall, '\n',
          "rouge1_f = ",rouge1_f, '\n',
          "rouge2_precision = ",rouge2_precision, '\n',
          "rouge2_recall = ", rouge2_recall, '\n',
          "rouge2_f = ", rouge2_f,'\n',
          "rougeL_precision = ", rougeL_precision, '\n',
          "rougeL_recall = ",rougeL_recall, '\n',
          "rougeL_f = ", rougeL_f, '\n',
          "sparse Vector ", sparse_accu)

    print("accuracy: ", sum(accuracy_vector) / len(accuracy_vector))

    print("Rouge1 Precision: ", sum(rouge1_precision)/len(rouge1_precision))
    print("Rouge1 Recall: ", sum(rouge1_recall) / len(rouge1_recall))
    print("Rouge1 FMeasure: ", sum(rouge1_f) / len(rouge1_f))

    print("Rouge2 Precision: ", sum(rouge2_precision) / len(rouge2_precision))
    print("Rouge2 Recall: ", sum(rouge2_recall) / len(rouge2_recall))
    print("Rouge2 FMeasure: ", sum(rouge2_f) / len(rouge2_f))

    print("RougeL Precision: ", sum(rougeL_precision) / len(rougeL_precision))
    print("RougeL Recall: ", sum(rougeL_recall) / len(rougeL_recall))
    print("RougeL FMeasure: ", sum(rougeL_f) / len(rougeL_f))


    print("Sparse Accuracy", sum(sparse_accu) / len(sparse_accu))

    print(datetime.datetime.now())
"""
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_samples',
        default=25000,
        type=int,
        help='maximum number of conversation pairs to use')

    parser.add_argument(
        '--max_length', default=40, type=int, help='maximum sentence length')

    parser.add_argument('--batch_size', default=50, type=int)

    parser.add_argument('--num_layers', default=2, type=int)

    parser.add_argument('--num_units', default=512, type=int)

    parser.add_argument('--d_model', default=256, type=int)

    parser.add_argument('--num_heads', default=8, type=int)

    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--activation', default='relu', type=str)

    parser.add_argument('--epochs', default=20, type=int)

    hparams = parser.parse_args()
    main(hparams)

