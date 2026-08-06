# Report Notes for Member E

## Related Work

Grad-CAM is used to visualise class-discriminative image regions in convolutional neural networks. It uses the gradients of a target class score flowing into a convolutional layer to weight that layer's activation maps. This produces a coarse heatmap showing which spatial regions contributed positively to the target class decision.

## Method

For C1, the analysis uses the frozen Scratch ResNet-18 checkpoint selected by validation macro-F1. The model is placed in evaluation mode and the same test preprocessing is used: Resize(256), CenterCrop(224), ToTensor, and ImageNet mean/std normalization.

The Grad-CAM target layer is `model.layer4[-1]`, the last residual block in ResNet-18. The target class is the model's predicted class, because the purpose is to explain the model decision. The same procedure is then applied to D's pretrained ResNet-18 on the same selected images.

## Case Selection

Three types of cases are analysed:

- Correct cases: high-confidence correct predictions.
- Error cases: high-confidence incorrect predictions.
- Confusion-pair cases: examples from the most frequent confusion pair, Imperial Eagle vs European Honey-buzzard.

The selected examples are recorded in `summaries/gradcam_case_summary.csv`.

## Results and Discussion Points

In correct cases, the Grad-CAM activation often covers the main organism or visually distinctive object region, such as the flower body or moth wings. This suggests that the model can use object-relevant visual evidence when the prediction is correct.

In error cases, the activation sometimes focuses on visually salient regions that are not sufficient for fine-grained classification. For example, the model may focus on a flower cluster or strong colour/texture cue while predicting the wrong species. These examples should be discussed as failure cases, not as proof that the model always ignores the organism.

In the Imperial Eagle vs European Honey-buzzard confusion pair, the activation is generally near the bird body or wings. This suggests the model is not simply using pure background, but the two species remain difficult because they are visually similar and belong to the same family.

D's pretrained ResNet-18 generally has stronger test performance than C1. The comparison figures should be used to discuss whether pretraining makes the activation more stable, more object-centred, or more discriminative in error and confusion cases.

EfficientNet-B0 is also included because it is D's best-performing pretrained model. It reaches higher test performance than the pretrained ResNet-18, so the three-model figures can be used to discuss whether the strongest classifier produces more object-centred Grad-CAM overlays. In some examples, EfficientNet-B0 corrects errors made by C1, while in other high-confidence error cases it can still focus on visually salient but insufficient cues.

When comparing ResNet-18 and EfficientNet-B0 heatmaps, note that their target layers are not architecturally identical: both ResNet-18 models use `model.layer4[-1]`, while EfficientNet-B0 uses `model.features[-1]`. Therefore, the comparison is qualitative rather than a perfectly layer-matched measurement.

## Limitation

Grad-CAM is an interpretability tool, not strict causal evidence. A highlighted region suggests where the model's class score is sensitive, but it does not prove that the model would necessarily change prediction if that region were removed or altered.
