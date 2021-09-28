# barlow-twins
A minimal implementation of https://arxiv.org/pdf/2103.03230.pdf.
We train a modified resnet18 on CIFAR10.

After one thousand epochs we achieve a test Acc@1:84.20%, Acc@5:99.32% with the KNN predictor on the self supervisedly trained embeddings.
We occasionaly observed NaNs when training with mixed precision and do not recommend its use, as it is not really necessary with the small model and small batch size.
