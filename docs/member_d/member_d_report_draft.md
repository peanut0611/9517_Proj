3\. Methods

3.1 Transfer Learning

The core idea of transfer learning is to transfer the hierarchical
visual knowledge learned from a source domain to a target domain. Under
the fine-tuning paradigm, the low- and mid-level feature extraction
capabilities learned by the source model during pre-training are
retained, while the original classification head is replaced with a new
500-dimensional output layer tailored to the target task. During
training, all network parameters are updated end-to-end using a
relatively small learning rate. The pre-trained weights provide the
model with strong general visual priors, including the ability to detect
fundamental edges, recognize colour transitions, and capture geometric
structures. These priors not only substantially reduce the target
task\'s dependence on large-scale annotated data and significantly
accelerate gradient convergence but also serve as a powerful form of
implicit regularization, effectively mitigating overfitting in
fine-grained image classification.

3.2 ResNet-18 （考虑一下，是否和C部分的2.1. Scratch
ResNet-18进行协调，让C部分单独写一个Scratch的内容，将Scratch和ResNet-18分开写）

ResNet-18 fundamentally transformed the conventional paradigm of
sequentially stacking convolutional layers by introducing Identity
Shortcut Connections. The network is constructed by stacking standard
residual blocks. Unlike traditional convolutional neural networks, which
attempt to learn the complete mapping from the input to the desired
output directly, residual networks require the convolutional layers
within each residual block to learn only the residual between the input
and the target mapping. This shortcut pathway provides an unobstructed
channel for gradient propagation throughout the network, allowing
gradients from deeper layers to be transmitted directly to shallower
layers during backpropagation. As a result, ResNet-18 effectively
addresses the vanishing gradient problem that commonly arises during the
training of deep neural networks, thereby ensuring stable optimization
and reliable convergence of deep architectures.

3.3 EfficientNet-B0

EfficientNet introduces a Compound Scaling strategy, which advocates
scaling the network depth, width, and input image resolution
simultaneously using a fixed set of scaling coefficients. This balanced
scaling approach enables the network to achieve optimal feature
representation while maintaining computational efficiency under limited
resource constraints. The fundamental building block of EfficientNet-B0
is the Mobile Inverted Bottleneck Convolution (MBConv), which combines
computationally efficient depth wise separable convolutions with the
Squeeze-and-Excitation (SE) channel attention mechanism. The SE module
explicitly models inter-channel dependencies by adaptively assigning
higher weights to more informative feature channels, allowing the
network to focus on discriminative local features that are crucial for
fine-grained recognition, such as the distinctive characteristics of a
bird\'s beak or the texture of a plant\'s leaf margins.

3.4 Ablation Studies（这部分可以考虑和C部分的2.2. Controlled
Augmentation Ablation合在一起写，有一些内容是有重复的）

To rigorously isolate the effect of individual variables on model
performance, a unified experimental environment was established. All
models were trained using an identical data pipeline, where input images
were resized to the same resolution and subjected to the same data
normalization and data augmentation strategies. Furthermore, all
experiments employed the same loss function to measure classification
error, the same gradient-based optimization algorithm, the same learning
rate scheduling strategy, and were trained under an identical number of
epochs and batch size. This controlled setup effectively eliminated the
influence of differences in training environments and optimizer
configurations, ensuring a fair comparison across all experiments.

When comparing transfer learning with training from scratch, the network
architecture was fixed as ResNet-18, and the training strategy and
optimization settings were kept identical. The only experimental
variable was the weight initialization method. This comparison was
designed to quantify the contribution of pre-trained generic visual
knowledge to convergence speed, resistance to overfitting,
classification accuracy, and the reduction of blind-spot classes.

When comparing different deep neural network architectures, the
fine-tuning strategy and all optimization settings were kept identical,
with only the backbone architecture being changed (ResNet-18 vs.
EfficientNet-B0). This experiment aimed to evaluate the relative
strengths and weaknesses of different architectures under the same
transfer learning framework in terms of fine-grained feature extraction
capability, class-balanced performance (macro-averaged F1), parameter
efficiency, and practical inference speed for real-world deployment.

4\. Experimental Results

4.1 Experimental
Setup（这段是关于超参数的设置，可以和其他人写的环境设置合一块，比如数据的处理）

The Stochastic Gradient Descent (SGD) optimizer with momentum was
adopted for all experiments. The momentum coefficient was set to 0.9 to
smooth the gradient update trajectory and accelerate convergence across
flat regions of the loss landscape. The weight decay coefficient was
fixed at 0.0001 to regularize the model by preventing excessive growth
of the network weights. The initial learning rate was uniformly set to
0.05 for all models. A Cosine Annealing learning rate scheduler was
employed, whereby the learning rate decreased smoothly according to a
cosine function as training progressed, without abrupt stepwise
reductions, eventually approaching zero by the 100th epoch. All models
were trained for a total of 100 epochs, and the batch size was fixed at
128, subject to hardware constraints. This ensured that every model
underwent the same number of weight update steps throughout a complete
training cycle, thereby guaranteeing a fair and consistent comparison
across all experiments.

4.2 ResNet-18 Pretrained

The ResNet-18 Pretrained model achieved a Top-1 accuracy of 65.10% on
the test set, with the Top-5 accuracy reaching 83.34%. The test loss was
1.6843. In terms of macro-level balanced performance, the model achieved
a Macro Precision of 0.6601, a Macro Recall of 0.6510, and a Macro
F1-Score of 0.6442. The close alignment among Macro Precision, Macro
Recall, and Macro F1-Score indicates that the network produced
relatively stable and balanced prediction outputs across different
classes. Benefiting from the effective gradient initialization provided
by the pre-trained weights, the total training time for 100 epochs was
reduced to 45.07 minutes (approximately 27 seconds per epoch), while the
test set inference time was only 11.63 seconds.

The training loss curve decreased rapidly and smoothly from an initial
value of approximately 4.5 to around 0.20. The validation loss dropped
sharply within the first 20 epochs and subsequently remained stable
around 1.65--1.75, without any signs of overfitting or performance
degradation throughout the training process. The validation Top-1
accuracy increased rapidly from an initial value of approximately 25%
and converged steadily at around 65%, while the Top-5 accuracy remained
consistently above 83%. These results demonstrate strong generalization
stability and highly effective gradient propagation performance of the
model.

![](media/image1.png){width="5.768055555555556in"
height="4.120138888888889in"}

Regarding confidence distribution and output calibration, the overall
average prediction confidence reached 0.7320. For correctly classified
samples (N = 3,255), the average confidence score increased to 0.8564,
with the distribution heavily concentrated toward the high-confidence
range of 0.95--1.0, forming a prominent peak of highly confident
predictions. In contrast, incorrectly classified samples (N = 1,745)
exhibited a reduced average confidence of 0.5003, and the peak of
incorrect predictions within the high-confidence region (\>0.8) remained
extremely low. These results indicate that the model demonstrates strong
certainty when making correct predictions, while maintaining relatively
cautious probability estimation when producing incorrect
classifications.![](media/image2.png){width="5.768055555555556in"
height="2.884027777777778in"}

The full confusion matrix exhibits a relatively clear and highly
continuous blue main diagonal, while the noisy patterns in the
non-diagonal regions have been significantly reduced. At the
macro-category level, ResNet-18 Pretrained achieves 100% accuracy for 40
classes (e.g., red pagoda, bear\'s head tooth, etc.). Analysis of its
most challenging subset of categories (Hardest Classes) shows that the
classes with the lowest accuracy are Golden-winged Warbler (20.0%),
Hartlaub\'s Gull (20.0%), and Ash-throated Flycatcher (20.0%). The
subset confusion matrix reveals that classification failures at this
stage are no longer caused by random misclassification but are instead
highly concentrated on fine-grained visual confusion among closely
related species within the same family. For example, Golden-winged
Warbler is mainly misclassified as Chestnut-backed Chickadee or American
Redstart, which share highly similar morphological characteristics and
plumage patterns. Similarly, Hartlaub\'s Gull is prone to confusion with
Black-headed Gull. These results indicate that ResNet-18 Pretrained
still suffers from insufficient feature extraction when dealing with
closely related species that differ only in subtle local
characteristics, such as slight variations in beak coloration or
fine-scale feather pattern differences.

![](media/image3.png){width="5.768055555555556in"
height="4.614583333333333in"}

![](media/image4.png){width="5.768055555555556in"
height="3.2958333333333334in"}

![](media/image5.png){width="5.768055555555556in"
height="2.1631944444444446in"}

4.3 EfficientNet-B0 Pretrained

EfficientNet-B0 Pretrained achieves a Top-1 accuracy of 76.66% and a
Top-5 accuracy of 90.50%, with a test loss of 1.0509. The model
demonstrates strong performance in macro-level evaluation metrics,
achieving a Macro Precision of 0.7763, a Macro Recall of 0.7666, and a
high Macro F1-Score of 0.7620. In terms of computational efficiency, due
to the MBConv modules incorporating depthwise separable convolutions and
channel attention mechanisms, the model requires approximately 79.11
minutes to complete 100 epochs of training (around 47.3s per epoch).
However, benefiting from its lightweight parameter design and optimized
network architecture, its inference time on the test set is only 5.99
seconds, demonstrating significant advantages for practical deployment
scenarios.

The training and validation curves demonstrate that EfficientNet-B0
Pretrained possesses excellent fitting capability and generalization
performance. The Train Loss smoothly converges from an initial value of
approximately 4.2 to around 0.15. After a rapid decline during the first
15 epochs, the validation loss (Val Loss) remains consistently within
the range of 1.10 to 1.20, exhibiting only minor fluctuations while
maintaining a stable downward trend throughout the training process. The
validation Top-1 accuracy increases smoothly and eventually stabilizes
within the range of 75%--76%, while the Val Top-5 accuracy remains
consistently around 90% over a long period. The curves of Macro
Precision, Macro Recall, and Macro F1-score show parallel and steady
increases without any noticeable oscillations, indicating outstanding
stability in the optimization process.

![](media/image6.png){width="5.768055555555556in"
height="4.120138888888889in"}

In terms of prediction confidence distribution and probability
calibration, EfficientNet-B0 achieves an overall average prediction
confidence of 0.8150. The correctly classified samples (N = 3,833) have
an exceptionally high average confidence of 0.9006, with the vast
majority of samples densely concentrated in the region where confidence
values approach 1.0, forming a prominent high-confidence peak on the
right side of the distribution. In contrast, the incorrectly predicted
samples (N = 1,167) have an average confidence of only 0.5343. Their
probability distribution is relatively uniform and flat, without
generating false predictions in any high-confidence region. This
demonstrates that the network provides highly reliable probability
estimation for its prediction outputs, indicating strong calibration
performance and robustness.

![](media/image7.png){width="5.768055555555556in"
height="2.884027777777778in"}

The full confusion matrix and weak-class analysis further validate the
model's advanced visual recognition capability. The 500×500-dimensional
full confusion matrix presents an extremely clear, continuous, and
deep-blue main diagonal, while the non-diagonal regions are almost
entirely white and transparent, indicating a relatively low overall
misclassification rate. In terms of fine-grained macro-level category
performance, EfficientNet-B0 Pretrained achieves 100% accuracy (Accuracy
= 1.0) for 81 classes (e.g., western horsenettle, Eastern Bark
Centipede, youth on age, Chestnut Bob, etc.).

By conducting an in-depth analysis of failure cases through the subset
confusion matrix, the primary sources of misclassification can be
categorized into two types. The first is extreme fine-grained confusion
among closely related plant or bird species within the same genus (e.g.,
prickly sowthistle being confused with the closely related plant
Broad-leaved Chervil). The second is caused by complex environmental
occlusion and background camouflage in wild scenes (e.g., Gray Wolf
being misclassified as Canada Lynx or Hooded Skunk under dense forest
and snowy background conditions). These findings demonstrate that, in
most cases, the model is capable of accurately capturing local visual
details through the SE attention mechanism. Misclassifications only
occur when the model encounters extreme cases involving highly similar
appearances and significant background overlap, highlighting its strong
robustness in fine-grained image recognition tasks.

![](media/image8.png){width="5.768055555555556in"
height="4.614583333333333in"}

![](media/image9.png){width="5.768055555555556in"
height="3.2958333333333334in"}

![](media/image10.png){width="5.768055555555556in"
height="2.1631944444444446in"}

5.Discussion

5.1 Transfer Learning vs. Training from Scratch (ResNet-18 Scratch vs.
Pretrained)

The comparison between ResNet-18 Scratch and ResNet-18 Pretrained
reveals the decisive impact of pretrained weights on model
generalization capability and optimization trajectories in fine-grained
image classification tasks. Experimental results demonstrate that by
only changing the weight initialization strategy---introducing
pretrained weights while maintaining the exact same network architecture
(ResNet-18), optimization algorithm, and number of training epochs---the
model's Top-1 accuracy achieves a significant improvement from 41.32% to
65.10%. Meanwhile, the Macro F1-Score increases substantially from
0.4045 to 0.6442, and the test loss is reduced by more than 48%.
Training from scratch caused the network to exhibit a "blind spot
phenomenon", with at least 15 classes achieving zero accuracy. In
contrast, Transfer Learning eliminated the zero-accuracy blind-spot
classes and increased the average confidence of correctly predicted
samples to 0. 8564.These results clearly demonstrate that, for
fine-grained image classification tasks with limited sample sizes or
many categories, loading pretrained weights plays a crucial role in
overcoming the model's generalization bottleneck.

5.2 Comparing Different Deep Neural Network Architectures (ResNet-18
Pretrained vs. EfficientNet-B0 Pretrained)

The results of ResNet-18 Pretrained and EfficientNet-B0 Pretrained
demonstrate that the EfficientNet-B0 Pretrained model achieves
comprehensive superiority over ResNet-18 Pretrained across all
performance metrics. The Top-1 accuracy is further significantly
improved from 65.10% to 76.66%, while the Top-5 accuracy exceeds 90.50%.
The Macro F1-Score increases from 0.6442 to 0.7620, and the number of
perfectly recognized classes with 100% accuracy rises dramatically from
40 to 81. Although EfficientNet-B0 requires a longer training time of
79.11 minutes compared with 45.07 minutes for ResNet-18, its test set
inference time is reduced substantially from 11.63 seconds to only 5.99
seconds. These results establish EfficientNet-B0 as a highly
cost-effective architecture for practical deployment in fine-grained
image classification applications.
