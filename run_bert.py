# coding=utf-8


import os
import time
import tensorflow_hub as hub

from bert import modeling
from bert import optimization
from bert import tokenization
from bert import run_classifier

import tensorflow as tf


def getPrediction(sentences, max_seq_length, tokenizer, estimator, label_list):
  default_label_for_test = label_list[0]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = default_label_for_test) for x in sentences]

  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=max_seq_length, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  res = []
  for sentence, prediction in zip(sentences, predictions):
      probabilities = prediction["probabilities"]
      top_x = 3
      top_args = probabilities.argsort()[-top_x:][::-1]
      pred_label = label_list[top_args[0]]
      confidence = sorted(probabilities)[-top_x:][::-1][0]
      res.append((sentence, pred_label, confidence))

  return res

def get_labels():
    return ['no_eat','eat', 'unknown']

def main(argv):
  [vocab_file, bert_config_file, \
          init_checkpoint, trained_checkpoint_dir, predict_batch_size] = argv
  tf.logging.set_verbosity(tf.logging.INFO)

  predict_batch_size = int(predict_batch_size)
  max_seq_length = 128
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  label_list = get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=False)

  run_config = tf.contrib.tpu.RunConfig(
      model_dir=trained_checkpoint_dir,
      )

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  use_tpu = False

  model_fn = run_classifier.model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=init_checkpoint,
      learning_rate=1e-5,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=False
      )

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=1,
      eval_batch_size=1,
      predict_batch_size=predict_batch_size)


  sentences = [
          "\u5348\u9910\u6ca1\u6709\u7740\u843d",
          "\u6211\u4e0d\u997f",
          "\u6211\u5403\u8fc7\u996d\u4e86",
          "\u5230\u6b64\u7ed3\u675f\u5566",
          "\u7d2f"
          ]
  predictions = getPrediction(sentences, max_seq_length, tokenizer, \
          estimator, label_list)
  for predict in predictions:
      print(predict)

  return 

  
if __name__ == "__main__":
  BERT_BASE_DIR="chinese_L-12_H-768_A-12"
  vocab_file = "{}/vocab.txt".format(BERT_BASE_DIR)
  bert_config_file = "{}/bert_config.json".format(BERT_BASE_DIR)
  #init_checkpoint = "{}/bert_model.ckpt".format(BERT_BASE_DIR)
  init_checkpoint = "model/model.ckpt-1437"
  trained_checkpoint_dir = "model"
  predict_batch_size = "1"
  tf.app.run(main=main, argv=[vocab_file, bert_config_file, \
          init_checkpoint, trained_checkpoint_dir, predict_batch_size])
