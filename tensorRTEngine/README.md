# RefineDet TensorRT inference
对tensorRT进行封装


## RefineDet inference实验结果（显卡为P4000）
`image size为320`

未使用tensorRT，batchsize=1的结果：29.46ms(voc模型)

| batchsize | time | 
| - | :-: | 
| 1 | 10.71 ms | 
| 2 | 19.30 ms | 
| 3 | 24.84 ms |
| 4 | 31.79 ms |

`image size为512`

未使用tensorRT，batchsize=1的结果：51.10ms(voc模型)

| batchsize | time | 
| - | :-: | 
| 1 | 21.43 ms | 
| 2 | 43.04 ms | 
| 3 | 57.40 ms |
| 4 | 74.25 ms |

其中数据均为使用相同的batchsize生成的tensorRT model进行inference得到的结果

而使用batchsize为4生成的tensorRT model，进行batchsize比4小的图片数量进行inference的结果会比上述方法慢1-2ms