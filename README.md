# DSC-uncertainty-quantification-evaluation
Evaluation of uncertainty quantification methods used in diagnosing skin cancer using deep learning

## Introduction
There currently exists various implementations of deep learning systems for use in diagnosing skin cancer from medical images – taking an image of a skin lesion and determining if the lesion is benign or malignant. 

However, for practical use in the medical field, providing a binary yes/no answer to whether a given image features an area showing skin cancer is not ideal. Instead, it is desirable for such a system to provide a score describing the likelihood that the given image contains an area affected by skin cancer. One approach to this problem is to calculate an “uncertainty score” – naturally, describing how uncertain the system is in the results of the image classification.

Across the deep learning field, there exists several distinct approaches for calculating an uncertainty score, however, until recently there has been no clear method to compare these different approaches against each other to evaluate which approach is best suited to a given problem.

2022-23 saw the publication of a paper from the Interactive Machine Learning Group [1] which provides a method for evaluating different approaches of quantifying uncertainty.

This project aims to use this new evaluation method to compare a variety of uncertainty quantification approaches with concern to diagnosing skin cancer using deep learning-based image classification systems to provide insight into which approaches are best suited to the task. 

## References
[1] Paul F. Jaeger, Carsten T. Lüth, Lukas Klein & Till J. Bungert “A CALL TO REFLECT ON EVALUATION PRACTICES FOR FAILURE DETECTION IN IMAGE CLASSIFICATION”, https://arxiv.org/pdf/2211.15259.pdf
