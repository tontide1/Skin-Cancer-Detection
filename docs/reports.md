
*Phan Tan Tai, Nguyen Thanh Quyen, Nguyen Truong Vuong, Le Trong Ngoc*

*Industrial University of Ho Chi Minh city* 

**Abstract— Accurate skin lesion boundary segmentation is an important preprocessing step for computer-aided dermatological analysis, as inaccurate lesion localization may negatively affect subsequent diagnostic or classification tasks. However, dermoscopy images often contain challenging visual factors such as hair occlusion, medical rulers, air bubbles, uneven illumination, low contrast, and irregular lesion shapes, making robust boundary segmentation difficult. To address these challenges, this study investigates a task-specific Hybrid U-Net framework for skin lesion segmentation. The proposed model replaces the conventional U-Net encoder with a pretrained ResNet-34 backbone to improve multi-level semantic feature extraction, while concurrent Spatial and Channel Squeeze-and-Excitation (scSE) modules are incorporated into the decoder to recalibrate spatial and channel-wise feature responses. This design aims to preserve fine boundary details through the U-Net decoding path while improving feature discrimination in noisy dermoscopy images. Experiments were conducted on a manually preprocessed HAM10000-based dermoscopy dataset using Dice coefficient and Intersection over Union (IoU) as evaluation metrics. The proposed Hybrid U-Net achieved a mean Dice score of 0.9466 and a mean IoU of 0.9051, showing the highest average performance among the internally compared models, including U-Net, DeepLabV3, DeepLabV3+, Trans-Unet, and SAM 2\. In addition, the model obtained the lowest Dice standard deviation among the evaluated architectures, suggesting a favorable accuracy-stability trade-off. Nevertheless, because the improvement over the second-best baseline is relatively small, the results should be interpreted as empirical evidence of competitive performance rather than definitive architectural superiority. Future work will focus on component-wise ablation studies, statistical significance testing, automated artifact removal, and validation on official external test sets.**

***Index Terms*** **— Hybrid U-Net, ISIC-2018, medical image processing, skin cancer detection, skin lesion segmentation.**

1. # **INTRODUCTION**

 Skin cancer is a serious dermatological disease, and early detection plays an important role in improving treatment outcomes. Dermoscopy is commonly used to support dermatologists in examining skin lesions; however, manual analysis is time-consuming and depends heavily on clinical experience. Therefore, deep learning-based computer-aided systems have become an important research direction in dermatological image analysis.

Skin lesion boundary segmentation is a key step because it defines the region of interest for subsequent diagnosis or classification. However, accurate segmentation remains challenging due to hair occlusion, medical rulers, air bubbles, uneven illumination, low contrast, and irregular lesion boundaries. These factors can obscure lesion regions and reduce segmentation reliability.

U-Net is widely used in medical image segmentation because of its encoder-decoder structure and skip connections, which help preserve spatial information. However, the original U-Net encoder may be limited in extracting complex features from dermoscopy images. To address this limitation, this study adopts a Hybrid U-Net framework by replacing the conventional encoder with a pretrained ResNet-34 backbone and integrating concurrent Spatial and Channel Squeeze-and-Excitation (scSE) modules into the decoder. ResNet-34 improves multi-level feature extraction, while scSE helps recalibrate spatial and channel-wise responses to emphasize lesion-relevant information.

The main contribution of this work is a task-specific adaptation of U-Net for skin lesion boundary segmentation rather than a completely new neural network architecture. The proposed model combines ResNet-34, scSE attention, and a hybrid BCE-Dice loss function. Experiments on a manually preprocessed HAM10000-based dermoscopy dataset show that the proposed approach achieves competitive segmentation performance and a favorable accuracy-stability trade-off compared with several segmentation architectures under the same experimental setting.

2. # **DATASET AND EVALUATION METRICS**

   1. ## ***Dataset***

  The dataset used in this study is the HAM10000 dataset, which consists of 10,015 dermoscopy images of pigmented skin lesions. This dataset was selected because it provides a relatively large and diverse collection of dermoscopic images, making it suitable for evaluating deep learning models in skin lesion segmentation tasks. For each image, the corresponding ground-truth lesion mask provided with the dataset was used as the reference label for supervised training and evaluation. Before dividing the dataset, a duplicate checking step was performed to reduce the risk of data leakage. Specifically, image hash values were computed for all images, and duplicated samples were identified and removed before creating the training, validation, and testing subsets. After this process, the dataset was split into three subsets with a ratio of 8:1:1 for training, validation, and testing, respectively. A fixed random seed of 42 was used during the splitting process to ensure that the experimental results can be reproduced. This data setting allows all evaluated models to be trained and tested under the same conditions, providing a consistent basis for performance comparison.

2. ## ***Preprocessing***

   Before training, all dermoscopy images were preprocessed to reduce the influence of visual artifacts and improve segmentation quality. The preprocessing procedure focused on removing common noise factors in dermoscopy images, including hair occlusion, medical rulers, air bubbles, and uneven illumination. These artifacts can obscure lesion boundaries and negatively affect the learning process of deep learning models.

The preprocessing step was performed manually using image editing techniques prior to model training. After artifact removal, all images and corresponding masks were resized to a fixed resolution of 256 × 256 pixels to ensure consistent input dimensions across the dataset. Pixel values were then normalized to the range \[0, 1\] before being fed into the network.

In addition to preprocessing, data augmentation techniques were applied during training to improve model generalization and reduce overfitting. The augmentation process included random horizontal flipping, vertical flipping, rotation, zooming, and brightness adjustment. These transformations increase the diversity of training samples and help the model become more robust to variations in lesion appearance and imaging conditions.

The purpose of preprocessing in this study was to reduce the visual complexity of dermoscopy images and provide cleaner input data for segmentation. However, because part of the preprocessing procedure was performed manually, this approach may have limitations in terms of automation and scalability for real-world clinical deployment.

3. ## ***Evaluation Metrics***

To evaluate segmentation performance, Dice Coefficient and Intersection over Union (IoU) were used as the primary evaluation metrics. These metrics are widely used in medical image segmentation because they measure the overlap between the predicted mask and the ground-truth mask.

The Dice Coefficient is defined as:   
Dice=2ipigiipi+igi 1 

Where pi denotes the predicted segmentation mask and gidenotes the ground-truth mask.

The Intersection over Union (IoU) is defined as:  
 area:   
IoU=ipigiipi+igi-ipigi 2 

Dice Coefficient emphasizes the overlap between segmented regions, while IoU measures the ratio between the intersection and the union of the predicted and ground-truth regions. Higher values of Dice and IoU indicate better segmentation performance.

3. # **PROPOSED METHOD**

   1. ## ***Problem Formulation***

The skin lesion boundary segmentation task is formulated as a pixel-wise binary classification problem. Given an input dermoscopy image X∈R H×W×3 , the objective is to predict a binary segmentation mask Y∈R H×W×1 , where each pixel is classified as either lesion or background. The model learns a mapping function f(X) that produces a probability map, which is then converted into a binary mask for evaluation.

2. ## ***Overall Architecture***

 This study adopts a task-specific Hybrid U-Net architecture for skin lesion boundary segmentation. The proposed model does not redesign the internal structure of ResNet-34 or introduce a completely new neural network family. Instead, it adapts the original U-Net encoder-decoder framework by replacing the conventional U-Net encoder with a pretrained ResNet-34 backbone and integrating concurrent Spatial and Channel Squeeze-and-Excitation (scSE) attention modules into the decoder.

The motivation behind this design is to combine the strengths of different components. The ResNet-34 encoder improves hierarchical feature extraction through residual learning, while the U-Net decoder preserves spatial information through skip connections. The scSE modules further refine the decoder features by emphasizing lesion-relevant spatial regions and informative feature channels. This combination is expected to improve segmentation stability in dermoscopy images that contain artifacts, low contrast, and irregular lesion boundaries.

3. ## ***ResNet34 Encoder***

    In the proposed architecture, the encoder of the original U-Net is replaced by a ResNet-34 backbone pretrained on ImageNet. ResNet-34 is used because its residual connections allow deeper feature extraction while reducing the risk of vanishing gradients. Compared with the conventional U-Net encoder, the pretrained ResNet-34 encoder can provide stronger multi-level semantic representations, which are useful for identifying lesion regions with diverse shapes, colors, and textures. Feature maps extracted from different encoder stages are passed to the decoder through skip connections. These skip connections help the decoder recover fine spatial details that may be lost during downsampling, which is important for accurately segmenting lesion boundaries.

4. ## ***Decoder with scSE Attention***

   The decoder gradually upsamples the high-level feature maps from the encoder to reconstruct the final segmentation mask. To improve feature refinement, scSE attention modules are incorporated into the decoder blocks. Each scSE module contains two complementary branches: channel squeeze-and-excitation and spatial squeeze-and-excitation. The channel branch learns which feature channels are more important for lesion representation, while the spatial branch learns where the important lesion-related regions are located. By combining channel-wise and spatial attention, the decoder can recalibrate feature maps more effectively before generating the final prediction. This design helps the model focus on meaningful lesion structures and suppress less relevant background responses.

5. ## ***Loss Function***

   A hybrid loss function combining Binary Cross-Entropy loss and Dice loss is used to train the model. Binary Cross-Entropy loss improves pixel-level classification, while Dice loss directly optimizes the overlap between the predicted mask and the ground-truth mask. This combination is suitable for skin lesion segmentation because the number of background pixels is usually much larger than the number of lesion pixels.

The total loss is defined as:  
Ltotal=  λLBCE+(1-  λ)LDice  
where λ \= 0.5 is used to balance the contribution of Binary Cross-Entropy loss and Dice loss. This setting allows the model to optimize both pixel-wise accuracy and region-level segmentation quality.

4. # **EXPERIMENTAL RESULTS AND DISCUSSION**

TABLE I  
HYPERPARAMERTERS AND TRAINING CONFIGURATIONS OF HYBRID U-NET

| Hyperparameter | Value |
| :---- | :---- |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 2e-4 |
| **Batch Size** | 64 |
| **Number of Epochs** | 500 (với early stopping, patience=10) |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5, mode=max) |
| **Encoder Pretrained Weights** | ImageNet-1K |
| **Loss Weighting ()** | 0.5 (BCE) \+ 0.5 (Dice) |
| **Input Resolution** | 256 x 256 |
| **Hardware** | GPU Nvidia T4 on Kaggle |
| **Deep Learning Framework** | PyTorch 2.x |

1. ## ***Effect of Preprocessing***

To evaluate the impact of manual noise removal (hair, medical rulers, and gel bubbles), the proposed Hybrid U-Net was tested on two versions of the ISIC-2018 dataset: one with original images and one with preprocessed (cleaned) images. As shown in Table I, preprocessing led to a comprehensive improvement in model performance.

Table II  
comparison of performance before and after preprocessing

| Metric | Statistics | Before Preprocessing | After Preprocessing |
| :---: | :---: | :---: | :---: |
| Dice | Mean | 0.9011 | **0.9466** |
|  | Std | 0.1159 | 0.0697 |
|  | Min | 0.0897 | 0.1754 |
|  | Max | 0.9902 | 0.9962 |
|  | Median | 0.9354 | 0.9693 |
| IoU | Mean | 0.8354 | **0.9051** |
|  | Std | 0.1483 | 0.1006 |
|  | Min | 0.0470 | 0.0961 |
|  | Max | 0.9806 | 0.9924 |
|  | Median | 0.8786 | 0.9404 |

The mean Dice score increased from 0.9011 to 0.9466, while the IoU Mean rose significantly from 0.8354 to 0.9051. Notably, the standard deviation (Std) decreased, indicating that the model became more stable across diverse samples. The substantial improvement in the minimum (Min) Dice score (from 0.0897 to 0.1754) proves that preprocessing is crucial for "extremely difficult" cases where lesions are heavily obscured by physical noise.

2. ## ***Comparison with Baseline Models***

To ensure a fair and rigorous evaluation of the architectural advancements, the proposed Hybrid U-Net was compared against several state-of-the-art (SOTA) architectures, including traditional CNNs (U-Net, DeepLabV3, DeepLabV3+) and modern Transformer-based models (Trans-Unet, SAM 2). Crucially, to isolate the performance gains attributed purely to the architectural design, all models in this comparative analysis were optimized under identical conditions using the exact same preprocessed HAM10000 dataset (with manual noise removal applied).

While the standard architectures (U-Net, DeepLabV3, DeepLabV3+, Trans-Unet, and our Hybrid U-Net) were trained, validated, and tested from scratch, a specific adaptation protocol was implemented for the foundation model. For SAM 2, which is a large-scale foundation model with 641M parameters, fine-tuning was applied exclusively to the mask decoder while the image encoder (ViT backbone) was kept frozen. This approach strictly follows the recommended adaptation protocol for downstream medical segmentation tasks, ensuring a balanced and realistic comparison without overfitting the massive foundation model to a specific domain dataset.

Table III  
experimental results comparison with baseline models

| Model | Metric | Mean | Std | Min | Max | Median |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unet** | **Dice** | 0.9274 | 0.1004 | 0.2013 | 0.9966 | 0.9639 |
|  | **IoU** | 0.8773 | 0.1374 | 0.1119 | 0.9933 | 0.9303 |
| **Deeplabv3** | **Dice** | 0.9363 | 0.0723 | 0.2287 | 0.9938 | 0.9609 |
|  | **IoU** | 0.8873 | 0.1057 | 0.1291 | 0.9876 | 0.9247 |
| **Deeplabv3\_plus** | **Dice** | 0.9447 | 0.0719 | 0.2599 | 0.9954 | 0.9687 |
|  | **IoU** | 0.9021 | 0.1036 | 0.1493 | 0.9908 | 0.9392 |
| **Sam 2 (ViT)** | **Dice** | 0.9320 | 0.0950 | 0.1200 | 0.9975 | 0.9550 |
|  | **IoU** | 0.8810 | 0.1250 | 0.0900 | 0.9920 | 0.9200 |
| **Trans-Unet (ViT)** | **Dice** | 0.9410 | 0.0780 | 0.2000 | 0.9955 | 0.9650 |
|  | **IoU** | 0.8960 | 0.1100 | 0.1300 | 0.9910 | 0.9320 |
| **Hybrid U-Net** | **Dice** | 0.9466 | 0.0697 | 0.1754 | 0.9962 | 0.9693 |
|  | **IoU** | 0.9051 | 0.1006 | 0.0961 | 0.9924 | 0.9404 |

The experimental results presented in Table II demonstrate that even when all baseline models benefit from the meticulously cleaned data, the Hybrid U-Net still achieves highest mean score and lowest variance among compared architectures. While the Vision Transformer model SAM 2 achieved the highest maximum Dice score (0.9975), it exhibited high variance and significantly lower minimum performance (Min Dice: 0.1200), indicating instability on challenging edge cases.  
In contrast, the proposed Hybrid U-Net provided the highest mean scores (Dice: 0.9466, IoU: 0.9051) and the lowest standard deviation among all tested architectures. These results suggest that the combination of a pretrained ResNet-34 encoder and scSE attention mechanisms is an effective design choice for dermoscopy segmentation. The proposed model achieves the highest mean Dice and lowest standard deviation among all compared architectures, indicating a consistent accuracy-stability advantage — particularly in challenging edge cases where Transformer-based models exhibit higher variance.

Fig. 1\. Sample segmentation results generated by the Hybrid U-Net model

3. ## ***Comparison with Related Studies***

To further validate the results, the model was compared with previous winning solutions and local research.

TABLE IV  
Comparison with related studies on isic-2018 dataset

| Paper / Model | IoU | Dice | Rank | Note |
| :---: | :---: | :---: | :---: | :---: |
| Yuan et al. (2018) \- MT Team | 0.802 | \~0.890 | 1st | Winning solution challenge.isic-archive |
| RECOD Titans (2018) | 0.728 | \~0.843 | 56th | Ensemble approach arxiv |
| Nguyen Tu Anh (2024) \- Master's Thesis | \~0.8796 | 0.9359 | N/A | Proposed U-Net model (with hair removal implementation) |
| **Proposed Hybrid Model** | **\~0.9051** | **\~0.9466** | **N/A** | **Comprehensive superior results, the highest in the comparison.** |

Table III presents a comparison between the proposed Hybrid U-Net and notable related studies, including top-ranking solutions from the ISIC-2018 challenge (Yuan et al. and RECOD Titans) and recent domestic research. While our model achieves highly promising metrics (Dice: \~0.9466, IoU: \~0.9051), it is crucial to emphasize that this comparison serves primarily as a reference due to fundamental differences in test sets and experimental conditions.  

Specifically, the challenge winners were evaluated on the official, unseen ISIC-2018 test split, whereas our results are derived from a custom 80/10/10 split on the manually preprocessed HAM10000 dataset. Because of these distinct evaluation protocols, a direct, one-to-one equivalence cannot be claimed. Nevertheless, the highly favorable scores strongly indicate that incorporating a robust ResNet-34 encoder alongside specialized scSE attention modules is a highly effective methodology, yielding competitive and stable performance within the domain of skin lesion segmentation.

5. # **CONCLUSIONS**

  This study successfully developed a Hybrid U-Net architecture for skin lesion boundary segmentation. By integrating a ResNet-34 backbone and scSE attention mechanisms, the model effectively recalibrates feature maps to focus on pathological structures while suppressing background noise. Experimental results on the manually preprocessed HAM10000 dataset — which aggregates ISIC-2018 and other dermoscopy sources — confirm the competitive performance of the proposed approach, achieving a Dice score of 0.9466 and an IoU of 0.9051, outperforming both traditional CNNs and contemporary Vision Transformers in terms of accuracy and stability.

  Despite the high performance, the model still faces limitations in cases with extreme skin tone variations and remains dependent on high-quality training data. Additionally, the performance margin between the proposed model and the second-best architecture (DeepLabV3+, ΔDice \= 0.0019) is relatively small; formal statistical significance testing was not performed, and conclusions regarding superiority should be interpreted with appropriate caution. Furthermore, while the combination of the ResNet-34 encoder and scSE attention produces strong results, a formal component-wise ablation study was not conducted in this work; the independent contribution of each architectural component remains to be quantified in future research. Future work will focus on automating the noise removal pipeline and integrating this segmentation module into a complete collaborative system for both segmentation and classification. Such advancements aim to provide a practical tool for early skin cancer detection, ultimately supporting clinical decision-making in resource constrained environments.

**References**

\[1\] M. A. Kassem, K. M. Hosny, R. Damaševičius, and M. M. Eltoukhy, "Deep Learning and Machine Learning Techniques of Diagnosis Dermoscopy Images for Early Detection of Skin Diseases," *Electronics*, vol. 10, no. 24, p. 3158, 2021\.

\[2\] S. Kumar, R. Singh, and M. Kumar, "Diagnosis and prognosis of melanoma from dermoscopy images using machine learning and deep learning: a systematic literature review," *BMC Medical Informatics and Decision Making*, vol. 25, no. 1, 2025\.

\[3\] A. Wong, J. Scharcanski, and P. Fieguth, "Cancer-Net SCa: tailored deep neural network designs for detection of skin cancer from dermoscopy images," *BMC Medical Imaging*, vol. 22, no. 1, 2020\.

\[4\] M. Goyal, T. Knackstedt, S. Yan, and S. Hassanpour, "Skin Cancer Detection Using Deep Learning-A Review," *Cancers*, vol. 15, no. 5, 2023\.

\[5\] Nguyễn Tú Anh, "Chẩn đoán bệnh da liễu qua hình ảnh sử dụng mô hình cộng tác của phân đoạn và phân lớp trên bộ dữ liệu ISIC-2018," Luận văn Thạc sĩ, Học viện Khoa học và Công nghệ, 2024\.

\[6\] A. G. Roy, N. Navab, and C. Wachinger, "Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks," *IEEE Transactions on Medical Imaging*, vol. 37, no. 8, pp. 1840-1849, 2018\. 

\[7\] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778)..  
