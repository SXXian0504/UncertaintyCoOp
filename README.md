# UncertaintyCoOp

UncertaintyCoOp has been accepted for presentation at ICME 2026.

**Abstract**

Multi-label image recognition typically assumes exhaustive annotations, whereas real-world datasets such as MS-COCO and PASCAL VOC often contain missing or incorrect labels. Treating unlabeled categories as negatives introduces false-negative supervision, leading to degraded performance under partial annotations. 
To address this issue, we propose Uncertainty-Guided Context Optimization (UncertaintyCoOp), a prompt learning framework that explicitly models prediction uncertainty for partial-label multi-label recognition.
UncertaintyCoOp consists of three components: (1) an entropy-confidence hybrid uncertainty estimator capturing epistemic and aleatoric uncertainty, (2) an uncertainty-guided prompt fusion mechanism combining positive, negative, and uncertain prompts, and (3) an uncertainty-aware loss with a momentum-updated teacher to ensure stable optimization under noisy supervision.
Extensive experiments on standard benchmarks demonstrate that UncertaintyCoOp achieves competitive performance while consistently enhancing robustness under partial annotations.

**Motivation**

![Framework](assets/Motivation_left_tag.png)

Illustration of incorrect and missing annotations in VOC and COCO datasets. Each example (a-d) compares the dataset ground truth (GT) with the actual image content (Actual). ✓ and × denote present and absent labels, respectively. Incorrect and missing annotations are highlighted in dark blue and gray dashed boxes, respectively.

**Framework**

![Framework](assets/framework_final.png)

Illustration of the proposed UncertaintyCoOp framework for partial-label MLR. For each category, a positive prompt, an uncertainty prompt, and a learnable negative embedding interact with image patch features to produce three directional predictions. The entropy-confidence uncertainty estimator generates an uncertainty coefficient, which adaptively fuses the multi-branch predictions and further guides the uncertainty-aware loss function with a reliability teacher distribution.
