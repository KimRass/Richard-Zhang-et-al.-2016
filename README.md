# 'Colorful Image Colorization' (Richard Zhang et al., 2016) (partial) implementation
## Paper Reading
- [Colorful Image Colorization](https://github.com/KimRass/Colorful-Image-Colorization/blob/main/papers/colorful_image_colorization.pdf)
## Empirical Probability Distribution
- On VOC2012 dataset
    - <img src="https://github.com/KimRass/Colorful-Image-Colorization/assets/67457712/5ea1c250-6f0d-4e8d-9135-81705b0b1d26" width="600">
<!-- ## Related Works
- Previous works have trained convolutional neural networks (CNNs) to predict color on large datasets. However, ***the results from these previous attempts tend to look desaturated.*** One explanation is that use loss functions that encourage conservative predictions. These losses are inherited from standard regression problems, where ***the goal is to minimize Euclidean error between an estimate and the ground truth.***
## Introduction
- Given the lightness channel $L$, our system predicts the corresponding $a$ and $b$ color channels of the image in the CIELAB colorspace. ***Predicting color has the nice property that training data is practically free: any color photo can be used as a training example, simply by taking the image’s L\* channel as input and its a\*b\* channels as the supervisory signal.***
## Architecture
- 'X': Spatial resolution of output
- 'C': Number of channels of output
- 'S': Computation stride, values greater than 1 indicate downsampling following convolution, values less than 1 indicate upsampling preceding convolution
- 'D': Kernel dilation
- 'Sa': Accumulated stride across all preceding layers (product over all strides in previous layers)
- 'De' Effective dilation of the layer with respect to the input (layer dilation times accumulated stride)
- 'BN' Whether BatchNorm layer was used after layer
- 'L': Whether a 1 x 1 conv and cross-entropy loss layer was imposed
## Training
### Loss
- Given an input lightness channel $X \in \mathbb{R}^{H \times W \times 1}$, our objective is to learn a mapping $\hat{Y} = F(X)$ to the two associated color channels $Y \in \mathbb{R}^{H \times W \times 2}$, where $H$, $W$ are image dimensions. (We denote predictions with a $\hat{\cdot}$ symbol and ground truth without.)
- We perform this task in CIELAB color space. Because distances in this space model perceptual distance, a natural objective function is the Euclidean loss $L_{2}$ between predicted and ground truth colors. However, ***this loss is not robust to the inherent ambiguity and multimodal nature of the colorization problem. If an object can take on a set of distinct a\*b\* values, the optimal solution to the Euclidean loss will be the mean of the set. In color prediction, this averaging effect favors grayish, desaturated results.*** Additionally, if the set of plausible colorizations is non-convex, the solution will in fact be out of the set, giving implausible results.
- Instead, we treat the problem as multinomial classification. We quantize the a\*b\* output space into bins with grid size 10 and keep the $Q = 313$ values.
- For a given input $X$, we learn a mapping to a probability distribution over possible colors $\hat{Z} \in [0, 1]^{H \times W \times Q}$, where $Q$ is the number of quantized a\*b\* values. To compare predicted $\hat{Z}$ against ground truth, we define function $Z = H^{-1}_{gt}(Y)$, which converts ground truth color $Y$ to vector $Z$, using a soft-encoding scheme.
- Soft-encoding scheme: Each ground truth value $Y_{h, w}$ can be encoded as a 1-hot vector $Z_{h, w}$ by searching for the nearest quantized a\*b\* bin. However, ***we found that soft-encoding worked well for training, and allowed the network to quickly learn the relationship between elements in the output space. We find the 5-nearest neighbors to*** $Y_{h, w}$ ***in the output space and weight them proportionally to their distance from the ground truth using a Gaussian kernel with*** $\sigma = 5$***.***
- Comment: 2-dimensional gaussian function ($\sigma = \sigma_{x} = \sigma_{y}$):
$$f(x, y) = \frac{1}{2\pi\sigma^{2}}\exp\bigg(-\frac{(x - \mu_{x})^{2} + (x - \mu_{y})^{2}}{2\sigma^{2}}\bigg)$$
- We then use multinomial cross entropy loss Lcl(·, ·), defined as: Lcl(Zb, Z) = − X h,w v(Zh,w) X q Zh,w,q log(Zbh,w,q) (2) where v(·) is a weighting term that can be used to rebalance the loss based on color-class rarity, as defined in Section 2.2 below. Finally, we map probability distribution Zb to color values Yb with function Yb = H(Zb), which will be further discussed in Section 2.3. 2.2 Class rebalancing The distribution of ab values in natural images is strongly biased towards val- ues with low ab values, due to the appearance of backgrounds such as clouds, pavement, dirt, and walls. Figure 3(b) shows the empirical distribution of pixels in ab space, gathered from 1.3M training images in ImageNet [28]. Observe that the number of pixels in natural images at desaturated values are orders of magnitude higher than for saturated values. Without accounting for this, the loss function is dominated by desaturated ab values. We account for the class- imbalance problem by reweighting the loss of each pixel at train time based on the pixel color rarity. This is asymptotically equivalent to the typical approach of resampling the training space [32]. Each pixel is weighed by factor w ∈ RQ, based on its closest ab bin.
### Dataset
- We train our network on the 1.3M images from the ImageNet training set, validate on the first 10k images in the ImageNet validation set.
## Evaluation
- Table 1: Colorization results on 10k images in the ImageNet validation set
    - <img src="https://user-images.githubusercontent.com/105417680/229296573-9e07c8cb-3f1f-4159-9d26-8e1c1fd3cd28.png" width="600">
    - Higher is better for all metrics.
    - 'Gray': Colors every pixel gray, with $(a^{*}, b^{*}) = 0$
    - 'Random': Random Copies the colors from a random image from the training set.
    - 'Larsson et al. [23]':
    - 'Ours (L2)': Our network trained from scratch, with L2 regression loss.
    - 'Ours (L2, ft)': Our network trained with L2 regression loss, fine-tuned from our full classification with rebalancing network.
    - 'Ours (class)': Our full method, with classification loss, defined in Equation 2, and class rebalancing, as described in Section 2.2. The network was trained from scratch with k-means initialization [36], using the ADAM solver for approximately 450k iterations3.
    - 'Ours (full)':  2. Ours (class) Our network on classification loss but no class rebalancing (λ = 1 in Equation 4).
    - 'AuC':
        - Refers to the area under the curve of the cumulative error distribution over a\*b\* space [22].
        - 'rebal' shows the class-balanced variant of this metric.
        - As a low-level test, ***we compute the percentage of predicted pixel colors within a thresholded L2 distance of the ground truth in a\*b\* color space. We then sweep across thresholds from 0 to 150 to produce a cumulative mass function, integrate the area under the curve (AuC), and normalize. Note that this AuC metric measures raw prediction accuracy, whereas our method aims for plausibility.***
        - Our network, trained on classification without rebalancing, outperforms our L2 variant (when trained from scratch) (Comment: 'AuC non-rebal'에 대해서 'Ours (class)'가 'Ours (L2)'보다 좋은 성능을 보입니다.).
        - When the L2 net is instead fine-tuned from a color classification network, it matches the performance of the classifica- tion network. This indicates that the L2 metric can achieve accurate coloriza- tions, but has difficulty in optimization from scratch. The Larsson et al. [23] method achieves slightly higher accuracy. Note that this metric is dominated by desaturated pixels, due to the distribution of ab values in natural images (Figure 3(b)). As a result, even predicting gray for every pixel does quite well, and our full method with class rebalancing achieves approximately the same score. Perceptually interesting regions of images, on the other hand, tend to have a distribution of ab values with higher values of saturation. As such, we compute a class-balanced variant of the AuC metric by re-weighting the pixels inversely by color class probability (Equation 4, setting λ = 0). Under this metric, our full method outperforms all variants and compared algorithms, indicating that class-rebalancing in the training objective achieved its desired effect.
    - 'VGG Top-1 Class Acc':
        - The classification accuracy after colorization using the VGG-16 [5] network.
        - Does our method produce realistic enough colorizations to be interpretable to an off-the-shelf object classifier? We tested this by feeding our fake colorized images to a VGG network [5] that was trained to predict ImageNet classes from real color photos. ***If the classifier performs well, that means the colorizations are accurate enough to be informative about object class.***
        - ***Classifier performance drops from 68.3% to 52.7% after ablating colors from the input.***
        - After re-colorizing using our full method, the performance is improved to 56.0% (other variants of our method (Comment: 'Ours (L2, ft)' and 'Ours (class)') achieve slightly higher results).
        - 'Larsson et al. [23]' achieves the highest performance on this metric, reaching 59.4%.
        - For reference, a VGG classification network fine-tuned on grayscale inputs reaches a performance of 63.5%.
        - In addition to serving as a perceptual metric, this analysis demonstrates a practical use for our algorithm: without any additional training or fine-tuning, we can improve performance on grayscale image classification, simply by colorizing images with our algorithm and passing them to an off-the-shelf classifier.
    - 'AMT Labeled Real':
        - For many applications, such as those in graphics, the ultimate test of colorization is how compelling the colors look to a human observer. To test this, we ran a real vs. fake two-alternative forced choice experiment on Amazon Mechanical Turk (AMT). Participants in the experiment were shown a series of pairs of images. Each pair consisted of a color photo next to a re-colorized version, produced by either our algorithm or a baseline. Participants were asked to click on the photo they believed contained fake colors generated by a computer program.
        - Shows results from our AMT real vs. fake test (with mean and standard error reported, estimated by bootstrap).
        - Note that an algorithm that produces ground truth images would achieve 50% performance in expectation. 
- 2. Semantic interpretability (VGG classification):
## Studies
- Figure 13
    - <img src="https://user-images.githubusercontent.com/105417680/229299407-d0129678-b0a4-44df-904d-2bbd4cc7900c.png" width="1000">
- To further investigate the biases in our system, we look at the common classification confusions that often occur after image recolorization, but not with the original ground truth image. Examples for some top confusions are shown in Figure 10. An image of a “minibus” is often colored yellow, leading to a misclassification as “school bus”. Animal classes are sometimes colored differently than ground truth, leading to misclassification to related species. Note that the colorizations are often visually realistic, even though they lead to a misclassification. To find common confusions, we compute the rate of top-5 confusion Corig, Crecolor ∈ [0, 1]C×C , with ground truth colors and after recolorization. A value of Cc,d = 1 means that every image in category c was classified as category d in the top-5. We find the class-confusion added after recolorization by computing A = Crecolor − Corig, and sort the off-diagonal entries. Figure 11(b) shows all C × (C − 1) off-diagonal entries of Crecolor vs Corig, with the top 100 entries from A highlighted. For each category pair (c, d), we extract the images that contained the confusion after recolorization, but

- color prediction requires understanding an image at both the pixel and the semantic-level. We have investigated how colorization generalizes to high-level semantic tasks in Section 3.2. Studies of natural image statistics have shown that the lightness value of a single pixel can highly constrain the likely color of that pixel: darker lightness values tend to be correlated with more saturated colors [44]. Could our network be exploiting a simple, low-level relationship like this, in order to predict color?4 We tested this hypothesis with the simple demonstration in Figure 12. Given a grayscale Macbeth color chart as input, our network was unable to recover its colors. This is true, despite the fact that the lightness values vary considerably for the different color patches in this image. On the other hand, given two recognizable vegetables that are roughly isoluminant, the system is able to recover their color. In Figure 12, we also demonstrate that the prediction is somewhat stable with respect to low-level lightness and contrast changes. Blurring, on the other hand, has a bigger effect on the predictions in this example, possibly because the operation removes the diagnostic texture pattern of the zucchini. E.g., previous work showed that CNNs can learn to use chromatic aberration cues to predict, given an image patch, its (x,y) location within an image [14].

- effective dilation. The effective dilation is the spacing at which consecutive elements of the convolutional kernel are evaluated, relative to the input pixels, and is computed by the product of the accumulated stride and the layer dilation. Through each convo- lutional block from conv1 to conv5, the effective dilation of the convolutional kernel is increased. From conv6 to conv8, the effective dilation is decreased.
## References
- [5] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [22] [Learning Large-Scale Automatic Image Colorization](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Deshpande_Learning_Large-Scale_Automatic_ICCV_2015_paper.pdf)
- [23] [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf)
# TO DO
## PyTorch Implementation
- Loss function 구현 -->
