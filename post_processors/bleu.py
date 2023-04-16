from typing import Dict, List, Any

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import distributed as dist
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
from tqdm import tqdm

from post_processors.dist_mixin import DistGatherMixin
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class BLEUMetric(DistGatherMixin):
    def __init__(self):
        self.predictions = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        sources = []
        targets = []
        for item in meta_data:
            sources.append(item["src"])
            if "tgt" in item and item["tgt"]:
                targets.append(item["tgt"])
            else:
                targets.append("")

        pred_seq = batch_model_outputs["generated_seq"]
        predictions = [
            {
                "source": src,
                "target": tgt,
                "prediction": pred,
            } for src, tgt, pred in zip(sources, targets, pred_seq)
        ]

        if ddp:
            obj = predictions
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

        del meta_data, batch_model_outputs, sources, targets, pred_seq, predictions

    def get_results(self):
        bleu = sum(
            [sentence_bleu([word_tokenize(pred["target"])], word_tokenize(pred["prediction"])) for pred in
             self.predictions]
        ) * 1.0 / len(self.predictions)

        return {"bleu": bleu}, self.predictions


def rouge_test(predictions):
    rouge = ROUGEScore()
    metric_score = {
        "rouge1_fmeasure": [],
        "rouge1_precision": [],
        "rouge1_recall": [],
        "rouge2_fmeasure": [],
        "rouge2_precision": [],
        "rouge2_recall": []
    }
    for pred in tqdm(predictions, total=len(predictions)):
        try:
            rs = rouge(pred["prediction"], pred["target"])
        except Exception as e:
            logger.error(f"Error in rouge calculation: {e}")
            continue
        metric_score["rouge1_fmeasure"].append(rs['rouge1_fmeasure'])
        metric_score["rouge1_precision"].append(rs['rouge1_precision'])
        metric_score["rouge1_recall"].append(rs['rouge1_recall'])
        metric_score["rouge2_fmeasure"].append(rs['rouge2_fmeasure'])
        metric_score["rouge2_precision"].append(rs['rouge2_precision'])
        metric_score["rouge2_recall"].append(rs['rouge2_recall'])

    metric_score["rouge1_fmeasure"] = float(np.array(metric_score["rouge1_fmeasure"]).mean())
    metric_score["rouge1_precision"] = float(np.array(metric_score["rouge1_precision"]).mean())
    metric_score["rouge1_recall"] = float(np.array(metric_score["rouge1_recall"]).mean())
    metric_score["rouge2_fmeasure"] = float(np.array(metric_score["rouge2_fmeasure"]).mean())
    metric_score["rouge2_precision"] = float(np.array(metric_score["rouge2_precision"]).mean())
    metric_score["rouge2_recall"] = float(np.array(metric_score["rouge2_recall"]).mean())

    # print("=== Evaluation score - Rouge score ===")
    # print("Rouge1 fmeasure:\t", metric_score["rouge1_fmeasure"])
    # print("Rouge1 precision:\t", metric_score["rouge1_precision"])
    # print("Rouge1 recall:  \t", metric_score["rouge1_recall"])
    # print("Rouge2 fmeasure:\t", metric_score["rouge2_fmeasure"])
    # print("Rouge2 precision:\t", metric_score["rouge2_precision"])
    # print("Rouge2 recall:  \t", metric_score["rouge2_recall"])
    # print("=====================================")
    return metric_score


class RougeMetric(DistGatherMixin):
    def __init__(self):
        self.predictions = []

    @staticmethod
    def clean_pred_seq(pred):
        output = []
        for word in pred:
            if word == "SOS":
                continue
            if word == "EOS":
                break
            output.append(word)
        return ' '.join(output)

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        sources = meta_data["input"]
        targets = meta_data["output"]
        indices = meta_data["index"]

        pred_seq = batch_model_outputs["generated_seq"]
        predictions = [
            {
                "source": src,
                "target": tgt,
                "prediction": self.clean_pred_seq(pred),
                "index": idx
            } for src, tgt, pred, idx in zip(sources, targets, pred_seq, indices)
        ]

        if ddp:
            obj = predictions
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self):
        predictions = []
        existing_ids = []
        for pred in self.predictions:
            if pred["index"] not in existing_ids:
                predictions.append(pred)
                existing_ids.append(pred["index"])

        metrics = rouge_test(self.predictions)

        return metrics, self.predictions
