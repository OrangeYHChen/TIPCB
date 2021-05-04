# TIPCB

### Prerequisites
* Pytorch 1.1
* cuda 9.0
* python 3.6
* GPU Memory>=12G

### Datasets
We evaluate our method on CUHK-PEDES. Please visit [Here](http://xiaotong.me/static/projects/person-search-language/dataset.html).

### Usage
* You need to generate tokens to "/data/BERT_encode/" by running "BERT_token_64.py" or downloading from [Here](https://drive.google.com/drive/folders/1gVWpGq7FJg6kSvK_wJH9BYQCikVHVydg?usp=sharing).
* If you want to train the network, you can run our code with the following commands:

``
python train.py --max-length 64 --batch-size 64 --num-epoches 80 --adam-lr 0.003 --gpus 0
``
* You can download our trained model from [Here](https://drive.google.com/file/d/1HjcXca9CGgRK6pUtKnHqy_Ad27u3VnY3/view?usp=sharing) and the trained log from [Here](https://drive.google.com/file/d/1IOMsRg_iXaquenraRyvLiRrdmR1e7oBt/view?usp=sharing).

### Evaluate
| Top-1 | Top-5 | Top-10 |
| :------: | :------: | :------: |
| 63.63 | 82.81 | 89.01 |
