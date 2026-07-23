# YOLOX-Tiny vs YOLO11n on Basketball

This compares the latest notebook runs:

- YOLO11n: `nb02-ultralytics_yolo11n_on_basketball.py`,
  run `outputs/runs/basketball/yolo11n_basketball-4`
- YOLOX-Tiny: `nb03-yolox_tiny_on_basketball.py`,
  run `outputs/runs/basketball/yolox_tiny_basketball`

Both models use 640x640 input and one class, `basketball`.

| Split | Images | Annotations | Background Images |
| --- | ---: | ---: | ---: |
| Train | 105 | 80 | 35 |
| Validation | 22 | 18 | 7 |
| Test | 23 | 18 | 8 |

## Model Size

| Model | Parameters | Layer / Module Count | GFLOPs |
| --- | ---: | ---: | ---: |
| YOLO11n | 2.590M | 182 layers | 6.4 |
| YOLOX-Tiny | 5.033M | 238 leaf modules / 363 total modules | 15.23 |

YOLOX-Tiny is about 1.94x larger by parameter count and 2.38x heavier by
GFLOPs. YOLO11n counts come from the Ultralytics summary. YOLOX-Tiny did not
print an Ultralytics-style layer summary, so its module counts were derived
from the local YOLOX model instance.

## Training And Validation

Best epoch is selected by validation `mAP50-95`.

| Model | Epochs | Training Time | Best Epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 100 | 25.59 min | 99 | 0.760 | 0.389 | 0.426 | 0.333 |
| YOLOX-Tiny | 150 | ~24.17 min | 92 | 0.471 | 0.444 | 0.439 | 0.332 |

Validation `mAP50-95` is essentially tied: 0.333 for YOLO11n and 0.332 for
YOLOX-Tiny. YOLO11n is much more precise; YOLOX-Tiny has slightly better
recall and `mAP50`.

YOLO11n training time comes from Ultralytics `results.csv`. YOLOX-Tiny
training time is timestamp-derived because its `results.csv` does not log
elapsed time.

## Held-Out Test

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11n | 0.846 | 0.609 | 0.652 | 0.399 |
| YOLOX-Tiny | 0.591 | 0.722 | 0.674 | 0.368 |

On test data, YOLO11n has better precision and `mAP50-95`. YOLOX-Tiny finds
more objects, giving it better recall and slightly better `mAP50`.

## Speed

All values are milliseconds per image.

| Model / Split | Preprocess | Inference / Forward | Loss | Postprocess / NMS | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n validation | 0.353 | 9.445 | 0.00004 | 11.745 | 21.544 |
| YOLO11n test | 0.473 | 197.155 | 0.00004 | 12.721 | 210.349 |
| YOLOX-Tiny validation | n/a | 11.548 | n/a | 0.735 | 12.283 |
| YOLOX-Tiny test | n/a | 60.545 | n/a | 0.763 | 61.308 |

Speed is useful directionally, but not a perfectly controlled benchmark:
Ultralytics reports preprocess, inference, loss, and postprocess, while the
YOLOX helper reports only forward and NMS.

## Summary

YOLO11n is the better default here: it is much smaller, lighter, and stronger
on held-out `mAP50-95`. YOLOX-Tiny is still useful as a recall-oriented
alternative when missing a basketball matters more than false positives.

## Top Improvements

The biggest limitation is data volume: the training split has only 105 images
and 80 labeled basketballs. The top three improvements include:

1. Add more labeled basketball instances.
   More varied positives should improve generalization and reduce overfitting.
2. Audit labels and add hard negatives.
   Clean labels reduce training noise; hard negatives reduce false positives.
3. Tune confidence and NMS thresholds.
   This matches the final model to the desired precision/recall tradeoff.
