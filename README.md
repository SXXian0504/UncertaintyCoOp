# UncertaintyCoOp

UncertaintyCoOp has been accepted for presentation at ICME 2026.

**Abstract**

Multi-label image recognition typically assumes exhaustive annotations, whereas real-world datasets such as MS-COCO and PASCAL VOC often contain missing or incorrect labels. Treating unlabeled categories as negatives introduces false-negative supervision, leading to degraded performance under partial annotations. 
To address this issue, we propose Uncertainty-Guided Context Optimization (UncertaintyCoOp), a prompt learning framework that explicitly models prediction uncertainty for partial-label multi-label recognition.
UncertaintyCoOp consists of three components: (1) an entropy-confidence hybrid uncertainty estimator capturing epistemic and aleatoric uncertainty, (2) an uncertainty-guided prompt fusion mechanism combining positive, negative, and uncertain prompts, and (3) an uncertainty-aware loss with a momentum-updated teacher to ensure stable optimization under noisy supervision.
Extensive experiments on standard benchmarks demonstrate that UncertaintyCoOp achieves competitive performance while consistently enhancing robustness under partial annotations.

