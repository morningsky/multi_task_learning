

## Multi-task Learning Models for Recommender Systems

This project is developed based on [DeepCTR](https://github.com/shenweichen/DeepCTR) :https://github.com/shenweichen/DeepCTR.

You can easy to use the code to design your multi task learning model  for multi regression or classification tasks.

## Quick Start

```python
from ple import PLE 

model = PLE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'regression'], 
            
task_names=['task 1','task 2'], num_levels=2, num_experts_specific=8, 

num_experts_shared=4, expert_dnn_units=[64,64], gate_dnn_units=[16,16], 
          
tower_dnn_units_lists=[[32,32],[32,32]])

model.compile("adam", loss=["binary_crossentropy", "mean_squared_error"], metrics=['AUC','mae'])

model.fit(X_train, [y_task1, y_task2], batch_size=256, epochs=5, verbose=2)

pred_ans = model.predict(X_test, batch_size=256)


```



### [Example 1](./example/demo.ipynb)

Dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/adult/

Task 1: (Classification) aims to predict whether the income exceeds 50K.

Task 2: (Classification) aims to predict this person’s marital status is never married.

### Example 2

Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/

*Preparing*

|               Model               |          Description           |                            Paper                             |
| :-------------------------------: | :----------------------------: | :----------------------------------------------------------: |
| [Shared-Bottom](shared_bottom.py) |         shared-bottom          | [Multitask learning](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf)(1998) |
|          [ESMM](essm.py)          | Entire Space Multi-Task Model  | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)(SIGIR'18) |
|          [MMoE](mmoe.py)          | Multi-gate Mixture-of-Experts  | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)(KDD'18) |
|         [CGC](ple_cgc.py)         |    Customized Gate Control     | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)(RecSys '20) |
|           [PLE](ple.py)           | Progressive Layered Extraction | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)(RecSys '20) |



## Shared-Bottom & MMOE



![mmoe&shared_bottom](https://laimc.oss-cn-shanghai.aliyuncs.com/blog/20210712231532.png)





## ESMM

![esmm1](https://laimc.oss-cn-shanghai.aliyuncs.com/blog/20210712231527.png)

##  CGC

![cgc](https://laimc.oss-cn-shanghai.aliyuncs.com/blog/20210712231607.png)

## PLE

![ple](https://laimc.oss-cn-shanghai.aliyuncs.com/blog/20210712231636.png)

