# Automated-Essay-Grading
An implementation of paper "[A Memory-Augmented Neural Model for Automated Grading](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=http%3A%2F%2Fdelivery.acm.org%2F10.1145%2F3060000%2F3053982%2Fp189-zhao.pdf%3Fip%3D218.94.142.61%26id%3D3053982%26acc%3DCHORUS%26key%3DBF85BBA5741FDC6E%252E180A41DAF8736F97%252E4D4702B0C3E38B35%252E6D218144511F3437%26__acm__%3D1576589244_38adbca828b48fe23ae695a24f78100d)" in PyTorch.

## Requirements
- PyTorch 1.2.0
- scikit-learn
- six
- python3

## Usage
```
# Train on essay set 1
python train.py --set_id 1
```
There are serval flags within train.py.
```
  --gpu_id        GPU_ID
  --set_id        SET_ID         essay set id, 1 <= id <= 8.
  --emb_size      EMB_SIZE       Embedding size for sentences.
  --token_num     TOKEN_NUM      The number of token in glove (6, 42).
  --feature_size  FEATURE_SIZE   Feature size.
  --epochs        EPOCHS         Number of epochs to train for.
  --test_freq     TEST_FREQ      Evaluate and print results every x epochs.
  --hops          HOPS           Number of hops in the Memory Network.
  --lr            LR             Learning rate.
  --batch_size    BATCH_SIZE     Batch size for training.
  --l2_lambda     L2_LAMBDA      Lambda for l2 loss.
  --num_samples   NUM_SAMPLES    Number of samples selected as memories for each score.
  --epsilon       EPSILON        Epsilon value for Adam Optimizer.
  --max_grad_norm MAX_GRAD_NORM  Clip gradients to this norm.
  --keep_prob     KEEP_PROB      Keep probability for dropout.
```
Better performance can be get by tuning hyper-parameters.

## Dataset
The dataset comes from Kaggle ASAP competition. You can download the data from [https://www.kaggle.com/c/asap-aes/data](https://www.kaggle.com/c/asap-aes/data).

Dataset details:

![avatar](./datainfo.png)

## Glove
Pre-trained word embeddings are used in this model. You can download `glove_42B_300d` from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/).