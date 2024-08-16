# Face Anti-Spoofing System with Generalization

This repository demonstrates the improvement of a baseline Face Anti-Spoofing (FAS) system using ResNet-50 by incorporating Gradient Reversal Layer (GRL) and Class-Conditional Domain Discriminator (CCDD) to create a more generalized model capable of handling multiple domain datasets.

## 1. Baseline Model
The baseline FAS model uses the ResNet-50 architecture, pre-trained on ImageNet, and fine-tuned to classify images as either 'real' or 'spoof'. This model, while effective on a single dataset, struggles to generalize across different domain datasets such as:
- CASIA-MFSD
- MSU-MFSD
- Oulu-NPU
- Idiap Replay-Attack (RA)

### Performance
The baseline model is evaluated on the above-mentioned domain datasets. However, it fails to perform well across domains, indicating poor generalization capability.

## 2. Generalized FAS Model with GRL and CCDD
To address the limitations of the baseline model, we introduced two key innovations:
- **Gradient Reversal Layer (GRL)**: This layer helps in learning domain-invariant features by reversing the gradient during backpropagation, thus confusing the domain classifier.
- **Class-Conditional Domain Discriminator (CCDD)**: This component helps in domain adaptation by distinguishing between domains based on the class label, ensuring that the model learns features that are not only domain-invariant but also class-specific.

### Performance
The generalized model is trained and evaluated on the same domain datasets. The GRL and CCDD innovations allow the model to better handle domain shifts, resulting in improved accuracy across multiple datasets.

## 3. Code Explanation
The code in this repository is split into two main parts:
1. **Baseline Model**: Implementation of the ResNet-50 based FAS model without domain adaptation.
2. **Generalized Model**: Implementation of the FAS model with GRL and CCDD for enhanced generalization across domains.

### Dependencies
- Python 3.7+
- PyTorch
- torchvision
- Other standard Python libraries (numpy, matplotlib, etc.)

### How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Run the Jupyter notebook `FAS_Generalized_Model_with_GRL_and_CCDD.ipynb` to train and evaluate the models.

## 4. Results
The results show that the generalized model significantly outperforms the baseline model in terms of accuracy across different domain datasets, showcasing the effectiveness of the proposed innovations.

## 5. References
- Paper: *Domain-Agnostic Feature Learning for Image and Video-Based Face Anti-Spoofing* by Saha et al.
