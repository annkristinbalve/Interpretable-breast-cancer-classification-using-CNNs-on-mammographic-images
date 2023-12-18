<img width="334" alt="image" src="https://github.com/annkristinbalve/Interpretable_Breast_Cancer_Classification/assets/76830039/18a0f0e8-9cbf-42b0-94ac-3db90545f4e6"># Bachelor Thesis in Cognitive Science and Artificial Intelligence at Tilburg University 
## Interpretable Breast Cancer Classification

In my Bachelor Thesis in explored the potential of Deep Learning methods for breast cancer classification. I focused on the application of Convolutional Neural Networks (CNNs) to classify normal, benign, and malignant breast tissue in mammograms. 
Additionally, I applied several post-hoc interpretability techniques to gain insights into the decision-making process of CNNs, shedding light on their 'black-box' predictions. 
Moreover, as part of this research, I created a new dataset of preprocessed mammograms which is available on Kaggle (https://lnkd.in/e3TYk7uG).

Research Goal: This thesis aims to investigate the interpretability
of Convolutional Neural Networks (CNNs) in mammogram classification, specifically focusing on understanding the underlying reasons
for the CNN’s predictions of breast cancer. By going beyond the
conventional accuracy evaluation and emphasizing interpretability,
this research addresses the crucial need to gain insights into the
decision-making process of CNNs. 

Methodology: The study utilizes
the Mammographic Image Analysis Society (MIAS) dataset which
is preprocessed to enhance image quality and extract Region of Interest (ROI) areas. Data augmentation is performed to address the
limited dataset size, and the training set is balanced to ensure equal
representation of all classes. After training a CNN to classify the
mammograms as normal, benign and malignant, three interpretability
techniques, namely LIME, Grad-CAM, and Kernel SHAP, are applied
and compared based on their computational efficiency, stability, and
quality of explanations. 

Results: The CNN model surpassed the baseline for all classes, but lacks ability to accurately classify the malignant
class, highlighting the need for improved detection of this critical
category. Among the interpretability techniques, Grad-CAM emerges
as the fastest and most robust algorithm, and generates heatmaps
resembling the abnormality shapes. 

Conclusion: Grad-CAM offers
comprehensive insights into the CNN’s behavior, identifying distinctive patterns in normal, benign, and malignant breast tissue. This
research demonstrates that extending a purely quantitative evaluation
of a CNN to post-hoc interpretability techniques can enhance the
understanding of CNN-based mammography classification.
