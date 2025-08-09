# barlow-twins
A minimal implementation of https://arxiv.org/pdf/2103.03230.pdf.
We train a modified resnet18 on CIFAR10.

After one thousand epochs we achieve a test Acc@1:84.20%, Acc@5:99.32% with the KNN predictor on the self supervisedly trained embeddings.
We occasionaly observed NaNs when training with mixed precision; AMP is now opt-in via `USE_AMP=1`.

## Performance tips
- Set `NUM_WORKERS` to `(CPU cores - 1)` to speed up data loading. Default now auto-detects.
- CUDA: tensors and model use `channels_last` memory format and AMP by default for faster throughput.
- Evaluation uses `torch.inference_mode()` to reduce overhead.
- Dataloaders enable `pin_memory`, `persistent_workers`, and `prefetch_factor` when appropriate.
