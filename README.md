# Cdiscount-Kaggle-Competition
15 million images - 5270 multi-class classification task!

Single submission scoring 69% on public and x% on private leaderboard.

Image classification challenge from France's largest e-commerce company.

```
Almost 9 million products: half of the current catalogue
More than 15 million images at 180x180 resolution
More than 5000 categories: yes this is quite an extreme multi-class classification!
```

Ran an InceptionV3 pretrained model for 30 epochs (lr=0.01) and then for 10 epochs (lr=0.001).

To run model:

`python bgen_InceptionV3.py`

To make a submission:

`python submit.py`
