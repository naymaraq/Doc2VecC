# Doc2VecC
PyTroch implemantation of Doc2VecC <br>
This repository contains workflows to reproduce Doc2VecC (see original paper [ICLR, 2017](https://openreview.net/pdf?id=B1Igu2ogg))

## Here are some website queries

```python
python3 nearest_neighbors.py -topk 10 -doc y -query auto.am
>>>Top 1 nearest: auto.am, score 1.0
>>>Top 2 nearest: auto.drom.ru, score 0.76
>>>Top 3 nearest: avtomobil.az, score 0.75
>>>Top 4 nearest: blog.sbtjapan.com, score 0.74
>>>Top 5 nearest: car.am, score 0.73
>>>Top 6 nearest: nomer.avtobeginner.ru, score 0.73
>>>Top 7 nearest: rosautopark.ru, score 0.72
>>>Top 8 nearest: forum.110km.ru, score 0.72
>>>Top 9 nearest: www.autobytel.com, score 0.72
>>>Top 10 nearest: philkotse.com, score 0.71
```
