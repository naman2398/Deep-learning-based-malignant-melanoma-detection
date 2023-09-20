# Skin Cancer Detection using Deep Learning

Skin Cancer, particularly Melanoma, has a very high fatality rate globally, and early diagnosis is crucial for successful treatment. This project aims to provide a solution to the challenging task of melanoma recognition through visual inspection of dermoscopy images. We have developed a two-stage deep learning technique that combines precise skin lesion segmentation with melanoma detection using convolutional neural networks (CNNs).

## Motivation

The motivation behind this project is to improve the accuracy and efficiency of melanoma detection. Visual inspection by clinicians is prone to errors due to the high resemblance between malignant and benign skin lesions. We wanted to harness the power of deep learning to create a more reliable and automated system for early melanoma detection.

## Problem Statement

Skin cancer, especially Melanoma, is highly curable when detected early. However, the similarity between malignant and benign samples in dermoscopy images makes accurate diagnosis challenging. This project aims to address this problem by leveraging deep learning to segment skin lesions and classify them for melanoma detection.

## What We Learned

During the course of this project, we gained valuable insights into the following areas:
- Deep learning techniques for image segmentation and classification.
- Handling and preprocessing of medical image datasets.
- The importance of integrating segmentation and classification for improved diagnostic performance.
- Collaborative efforts in the field of medical image analysis.

## Key Features

- Two-stage approach: Skin lesion segmentation followed by melanoma detection.
- Utilizes the DoubleU-Net architecture for precise lesion boundary segmentation.
- Employs five different CNN architectures ([DenseNet201](link1), [ResNet152V2](link2), [EfficientNetB7](link3), [InceptionV3](link4), and [InceptionResnetV2](link5)) for melanoma classification.
- Enhances diagnostic performance by focusing on segmented lesions, not the entire dermoscopy image.

## Dataset

Our deep learning model was trained and evaluated on three open-source independent datasets provided by the "International Skin Imaging Collaboration (ISIC) Melanoma Project":
- [ISIC 2017](https://challenge.isic-archive.com/data/#2017): Used for training the segmentation model as it contains skin lesion images with binary masks.
- [ISIC 2018](link7) and [ISIC 2019](link8): Utilized for classification.
- Duplicates have been removed, resulting in a final dataset with segmented images categorized into eight class labels:
  1. Melanocytic Nevus (NV)
  2. Benign Keratosis (BKL)
  3. Melanoma (MEL)
  4. Actinic Keratosis (AK)
  5. Vascular Lesion (VASC)
  6. Squamous Cell Carcinoma (SCC)
  7. Basal Cell Carcinoma (BCC)
  8. Dermatofibroma (DF)
  
  It's important to note that Melanoma (MEL) is cancerous, while the other labels are non-cancerous.

## Getting Started

To get started with this project, you can follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies as mentioned in the project's documentation.
3. Download the dataset and follow the data preprocessing steps.
4. Train and evaluate the deep learning model using the provided scripts.
5. Explore the model's performance and contribute to its improvement.

## Contribution

We welcome contributions from the community to enhance the accuracy and capabilities of our skin cancer detection system. Feel free to open issues, submit pull requests, or reach out to us for collaboration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is intended for research and educational purposes. It should not replace professional medical advice or diagnosis. If you have concerns about your skin health, consult a healthcare professional.
