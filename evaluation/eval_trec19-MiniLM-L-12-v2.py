import sys, math, tqdm, json, pytrec_eval, gzip, os, tarfile, logging
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample
from transformers import AutoTokenizer
# from google.colab import drive
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
"""# Initializing variables
"""
model_name = "/ivi/ilps/personal/aaskari/minilmv3/finetuned_CEs/final_models/ms-marco-MiniLM-L-12-v2-v2/"
fine_tuned_model_path = model_name
ranking_output_path = model_name + "trec19.ranking"
base_write_path = "/ivi/ilps/personal/aaskari/minilmv3/"
base_path = base_write_path + "msmarco-data/"
qrel_path = base_path + "2019qrels-pass.txt"
top100_run_path = base_path + "msmarco-passagetest2019-top1000.tsv"
queries_path = base_path + "msmarco-test2019-queries.tsv"
global_min_bm25 = 0
global_max_bm25 = 50
pos_neg_ratio = 4
max_train_samples = 0 # full train set
valid_max_queries = 0 # full validation set
valid_max_negatives_per_query = 0 # full negatives per query
corpus_path = base_path + "collection.tsv"
triples_train_path = base_path + "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
triples_validation_path = base_path + "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv"
max_length_query = 30
max_length_passage = 200
model_max_length = 230 + 3 + 3 # 3:[cls]query[sep]doc[sep]. Because we do have injecting into the input, we do not consider 3 token for v2.1! 3 extra tokens are needed in injection because bm25 score: normally takes two tokens, and onre more sep bm25 score [sep]

print("fine_tuned_model_path {} | model_max_length {} | queries_path {} | ranking_output_path {} ".format(fine_tuned_model_path, model_max_length, queries_path, ranking_output_path))

"""# CrossEncoder Class

"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                 default_activation_function = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes = True) # ignore_mismatched_sizes = True for transfer learning. first post_training, then using it for binary classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            accumulation_steps: int = 1,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param accumulation_steps: Number of steps to accumulate before performing a backward pass
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in tqdm.trange(epochs, desc="Epoch"):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()
            for i, (features, labels) in tqdm.tqdm(enumerate(train_dataloader), total=(len(train_dataloader) // accumulation_steps), desc="Iteration", smoothing=0.05):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                        loss_value /= accumulation_steps

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value /= accumulation_steps
                    loss_value.backward()
                    if (i + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                if not skip_scheduler and (i + 1) % accumulation_steps == 0:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)



    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm.tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)



"""# Evaluator Class"""

import numpy as np
import os
import csv
import pytrec_eval
import tqdm
from sentence_transformers import LoggingHandler, util
class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, ndcg@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, all_metrics: set = {"recall.1"}, name: str = '', write_csv: bool = True, show_progress_bar: bool = False):
        self.samples = samples
        self.name = name
        self.all_metrics = all_metrics

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps"] + list(all_metrics)
        self.write_csv = write_csv
        self.mean_metrics = {}
        self.show_progress_bar = show_progress_bar
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        mean_ndcg = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_ndcg_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        run = {}
        qrel = {}
        print("len: self.samples: " + str(len(self.samples)))
        try:
            for instance in tqdm.tqdm(self.samples):
                # print("instance: ", instance)
                qid = instance['qid']
                query = instance['query']
                positive = list(instance['positive'])
                negative = list(instance['negative'])
                positive_pids = list(instance['positive_ids'])
                negative_pids = list(instance['negative_ids'])
                docs =  negative + positive
                docs_ids = negative_pids + positive_pids
                is_relevant = [False]*len(negative) +  [True]*len(positive)

                qrel[qid] = {}
                run[qid] = {}
                for pid in positive_pids:
                    qrel[qid][pid] = 1

                if len(positive) == 0 or len(negative) == 0:
                    continue

                num_queries += 1
                num_positives.append(len(positive))
                num_negatives.append(len(negative))

                model_input = [[query, doc] for doc in docs]
                if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
                    pred_scores = model.predict(model_input, apply_softmax=True, batch_size=16, show_progress_bar = self.show_progress_bar)[:, 1].tolist()
                else:
                    pred_scores = model.predict(model_input, batch_size=16, show_progress_bar = self.show_progress_bar).tolist()
                for pred_score, did in zip(list(pred_scores), docs_ids):
                    line = "{query_id} Q0 {document_id} {rank} {score} STANDARD\n".format(query_id=qid,
                                                                                          document_id=did,
                                                                                          rank="-10",#rank,
                                                                                          score=str(pred_score))
                    run[qid][did] = float(pred_score)

            evaluator = pytrec_eval.RelevanceEvaluator(qrel, self.all_metrics)
            scores = evaluator.evaluate(run)
            self.mean_metrics = {}
            metrics_string = ""
            for metric in list(self.all_metrics):
                self.mean_metrics[metric] = np.mean([ele[metric.replace(".","_")] for ele in scores.values()])
                metrics_string = metrics_string +  "{}: {} | ".format(metric, self.mean_metrics[metric])
            print("metrics eval: ", metrics_string)
            logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        except Exception as e:
            logger.error("error: ", e)
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f: # early stopping can be done by modifying this part. You can read this csv file. Then you need to count: best_step - last step + 1. if it is >earlystopping. then, you can just do sys.exit(1) to kill the process :)
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, sum(self.mean_metrics.values())])
        return sum(self.mean_metrics.values())

"""#Data

## utils

### read collections
"""

def read_collection(f_path):
  corpus = {}
  with open(f_path, "r") as fp:
    for line in tqdm.tqdm(fp, desc="reading {}".format(f_path)):
      did, dtext = line.strip().split("\t")
      corpus[did] = dtext
  return corpus
from glob import glob

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side = "right") # right is default btw.

def get_truncated_dict(id_content_dict, tokenizer, max_length):
  for id_, content, in tqdm.tqdm(id_content_dict.items()):
    truncated_content = tokenizer.batch_decode(tokenizer(content, padding=True, truncation=True, return_tensors="pt", max_length=max_length)['input_ids'], skip_special_tokens=True)[0]
    id_content_dict[id_] = truncated_content
  return id_content_dict

# Read our training file
"""### reading top1000: utils"""

def read_top1000_run(f_path, corpus, queries, separator = " "):
  samples = {}
  with open(f_path, "r") as fp:
    for line in tqdm.tqdm(fp, desc="reading {}".format(f_path)):
      # qid, _, did, rank, score, __ = line.strip().split(separator)
      qid, pid, query, passage = line.strip().split("\t")
      if qid not in queries: continue
      query = queries[qid]
      if qid not in samples:
        samples[qid] = {'qid': qid , 'query': "{} [SEP] {}".format(scores[qid][did], query), 'docs': list(), 'docs_ids': list()}
      samples[qid]['docs'].append(corpus[did])
      samples[qid]['docs_ids'].append(did)
  return samples


"""## Reading data


"""### reading qrel"""

with open(qrel_path, 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

### reading corpus and queries and truncate it 

queries = read_collection(queries_path)
corpus =  read_collection(corpus_path)

queries = get_truncated_dict(queries, tokenizer, max_length_query)
corpus = get_truncated_dict(corpus,tokenizer, max_length_passage)


"""### reading top1000: main"""
test_samples = read_top1000_run(top100_run_path, corpus, queries, separator = " ")



"""# Evaluating"""

# model = CrossEncoder(fine_tuned_model_path, num_labels=1, max_length=model_max_length)
model = CrossEncoder(fine_tuned_model_path, num_labels=1, max_length=model_max_length)

evaluator = CERerankingEvaluatorTest(
    test_samples,
    qrel,
    all_metrics = {"ndcg_cut.10", "map_cut.1000", "recall.10"},
    ranking_output_path = ranking_output_path,
    batch_size = batch_size
)
measures_results = evaluator.rank(model)

print("measures_results: ", measures_results)

