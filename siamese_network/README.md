# Siamese Network
Siamese network is a neural net architecture that takes in an input pair, propagates them through the same network with the same weights, and determine if the two inputs are similar (i.e. of the same class). It is often used in one-shot learning where there is only one or very few training examples from a class, for example facial recognition. The model attempts to learn the weights in a way that can best cluster similar inputs together while differentiate those of different classes. Siamese networks are a great tool to learn embeddings when training examples are sparse.

In this work, I implemented a siamese network with the MNIST hand-written digits dataset using a network with 2 convolutional layers and 2 fully connected layers. Contrastive loss function was used to compare the embedding pairs. Triplet loss may also be used as an alternative which will be future work.

A regular CNN of similar arthitecture is also implemented on the MNIST dataset to compare the learned embeddings with the siamese network. Because the embeddings are of dimensional 2 while the CNN classifier needs 10 outputs for classification, I designed the CNN so that before the 10-dimensional output layer, there is a 2-dimensional embedding layer with no non-linearities between them. This way, we can visualize the embeddings in 2D with a simple classifier such as logistic regression.

## Results
It can be seem from the plots below that:
* The embeddings from the siamese network are better at seperating different classes. This is in part because Euclidean distance is used for class differentiation in the contrastive loss.
* The embeddings from the CNN also achieves the same purpose as a simple linear classifier (logistic regression in this case) can do a fair job in classifying them. However, the embeddings are inherently different compared to those from the siamese network, and from a L2-norm perspective they are not as good. 
* It is slower to train the siamese network for each epoch. This is because 1) each training sample involves 2 images as input which means twice the computation, and 2) it takes some time to load the training examples -- to prevent class imblance in training we had to sample the input pairs such that there is a 50/50 chance the pairs are from the same class. This sampling step needs additional computation as well.
* The siamese network is **faster** at differentiating the classes. Visually, looking at the plots before for Epoch 2 and Epoch 5, it is evident that the clusters are more distinct in the siamese network versus the CNN.

|   | CNN | Siamese |
| ------------- | ------------- | ------------- |
| Before Training| <p align="center"> <img src=".//results_cnn/Epoch_0.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_0.png" width="400"/> </p> |
| Epoch 1| <p align="center"> <img src=".//results_cnn/Epoch_1.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_1.png" width="400"/> </p> |
| Epoch 2| <p align="center"> <img src=".//results_cnn/Epoch_2.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_2.png" width="400"/> </p> |
| Epoch 5| <p align="center"> <img src=".//results_cnn/Epoch_5.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_5.png" width="400"/> </p> |
| Epoch 10| <p align="center"> <img src=".//results_cnn/Epoch_10.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_10.png" width="400"/> </p> |
| Epoch 20| <p align="center"> <img src=".//results_cnn/Epoch_20.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_20.png" width="400"/> </p> |
| Epoch 50| <p align="center"> <img src=".//results_cnn/Epoch_50.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Epoch_50.png" width="400"/> </p> |
| Log Reg | <p align="center"> <img src=".//results_cnn/Simple Classifier.png" width="400"/> </p>  | <p align="center"> <img src=".//results_siamese/Simple Classifier.png" width="400"/> </p> |


Credits to: \
https://github.com/leimao/Siamese_Network_MNIST \
https://github.com/adambielski/siamese-triplet/blob/master/datasets.py

References: \
https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf \
https://medium.com/@subham.tiwari186/siamese-neural-network-for-one-shot-image-recognition-paper-analysis-44cf7f0c66cb \
https://weiminwang.blog/2019/03/01/whale-identification-5th-place-approach-using-siamese-networks-with-adversarial-training/

