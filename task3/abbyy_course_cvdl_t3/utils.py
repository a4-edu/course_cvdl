import json
from typing import List
from pathlib import Path
import numpy as np
from abbyy_course_cvdl_t3.coco_text import COCO_Text
from abbyy_course_cvdl_t3 import coco_evaluation


def dump_detections_to_cocotext_json(*,
    image_ids: List[int],
    xlefts:List[float],
    ytops:List[float],
    widths: List[float],
    heights: List[float],
    scores: List[float],
    path: str):
    """
    Сохраняет детекции detections в json-файл path в формате.
    Формат файла такой же, какой используют авторы в репозитории
    https://github.com/andreasveit/coco-text.
    Принимает все атрибуты детекций как списки.
    """
    path = Path(path)
    assert path.parent.exists(), f"path {path.parent} does not exist"
    detection_lists = [
        image_ids,
        xlefts,
        ytops,
        widths,
        heights,
        scores,
    ]
    assert all(isinstance(z, list) for z in detection_lists), "Detections params must be passes as lists"
    length = len(image_ids)
    assert all(len(z) == length for z in detection_lists), "Detections params must have the same length"

    prepared_detections = []
    for idx in range(length):
        prepared_detections.append({
            "image_id": image_ids[idx], "score": scores[idx],
            "bbox": [xlefts[idx], ytops[idx], widths[idx], heights[idx]],
            "utf8_string": "", "category_id": 0
        })

    with open(path, "w") as f:
        f.write(json.dumps(prepared_detections))
    return prepared_detections


def interpolated_average_precision_from_pr(recall_list, precision_list):
    """
    AP@IoU=T - это площадь под Precision-Recall кривой (в зависимости от confidence),
    где матч между детекциями сделан по порогу IoU=T.
    AP = sum (R[n+1] - R[n]) * Pi[n+1]
    """
    ap = 0.0
    assert len(precision_list) == len(recall_list)
    inv_precision_list = precision_list[::-1]
    inv_recall_list = recall_list[::-1]
    p_inter = -1
    prev_rec = 1.0
    for prec, rec in zip(precision_list, recall_list):
        if prec > p_inter:
            p_inter = prec
        ap += p_inter * (prev_rec - rec)
        prev_rec = rec
    return ap


def evaluate_ap_from_cocotext_json(*,
    coco_text: COCO_Text,
    path: str,
    area_fraction_threshold=1./32/32
    ):
    """
    Вычисляет average precision для json файла детекций.
    Возвращает AP, precisions для разных порогов и recalls для разных
    порогов score.
    """
    assert isinstance(coco_text, COCO_Text), "coco_text must be COCO_Text ( surprize !)"
    assert Path(path).exists(), f"{path} does not exist"

    results = coco_text.loadRes(str(path))
    precisions = []
    recalls = []

    for score_thr in np.arange(0, 1, 0.1):
        matches = coco_evaluation.getDetections(
            coco_text,
            results,
            imgIds=coco_text.val,
            score_threshold=score_thr,
            area_fraction_threshold=area_fraction_threshold
        )
        tp = len(matches['true_positives'])
        fp = len(matches['false_positives'])
        fn = len(matches['false_negatives'])
        eps = 1e-6
        precisions.append(
            tp / (tp + fp + eps)
        )
        recalls.append(
            tp / (tp + fn + eps)
        )
    ap = interpolated_average_precision_from_pr(recalls, precisions)
    return ap, precisions, recalls

