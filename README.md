# Machine Learning 2019 SPRING

NTUEE Course
<br/>

## HW1

### PM2.5 Prediction
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw1)
```
bash hw1.sh <input_file> <output_file>
bash hw1_best.sh <input_file> <output_file>
```
- `<input_file>` is the path to `test.csv`
- `<output_file>` is the path to the output prediction file (e.g. `ans.csv`)
<br/>

|          | Public   | Private  |
| :------: | :------: | :------: |
| RMSE     | 5.46876  | 7.06495  |
<br/>


## HW2

### Income Prediction
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw2)
```
bash hw2_logistic.sh $1 $2 $3 $4 $5 $6
bash hw2_generative.sh $1 $2 $3 $4 $5 $6
bash hw2_best.sh $1 $2 $3 $4 $5 $6
```
- `$1` is the path to `train.csv`
- `$2` is the path to `test.csv`
- `$3` is the path to `X_train.csv`
- `$4` is the path to `Y_train.csv`
- `$5` is the path to `X_test.csv`
- `$6` is the path to the output prediction file (e.g. `ans.csv`)
<br/>

|          | Public   | Private  |
| :------: | :------: | :------: |
| Accuracy | 0.85884  | 0.85861  |
<br/>


## HW3

### Image Sentiment Classification
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw3)
```
bash hw3_train.sh <training_data>
bash hw3_test.sh <testing_data> <output_file>
```
- `<training_data>` is the path to  `train.csv`
- `<testing_data>` is the path to `test.csv`
- `<output_file>` is the path to the output prediction file (e.g. `ans.csv`)
<br/>

|          | Public   | Private  |
| :------: | :------: | :------: |
| Accuracy | 0.69768  | 0.69629  |
<br/>


## HW4

### Explain Your Model
*Same data as HW3. No Kaggle.*
```
bash hw4.sh <training_data> <output_dir/>
```
- `<training_data>` is the path to `train.csv`
- `<output_dir/>` (e.g. `output/`) is the directory to output
    - `fig1_{0,1,2,3,4,5,6}.jpg` (one image for each label class)
    - `fig2_1.jpg`
    - `fig2_2.jpg`
    - `fig3_{0,1,2,3,4,5,6}.jpg` (one image for each label class)
<br/>


## HW5

### Adversarial Attack
*No Kaggle.*
[Data](https://drive.google.com/file/d/1fwBlgnB64Jl8QAFXUGq-oGP6e42iMo6f/view)
```
bash hw5_fgsm.sh <input_img_dir> <output_img_dir>
bash hw5_best.sh <input_img_dir> <output_img_dir>
```
- `<input_img_dir>` is the directory of 200 original input images (e.g. `./images`)
- `<output_img_dir>` is the directory to output 200 adversarial output images (e.g. `./output`)
<br/>

| Success rate | L-Inifinity |
| :----------: | :---------: |
| 0.995        | 3.0000      |
<br/>


## HW6

### Malicious Comments Identification
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw6/)
```
bash hw6_test.sh <test_x> <dict.txt.big> <output_file>
```
- `<test_x>` is the path to `test_x.csv`
- `<dict.txt.big>` is the path to `dict.txt.big` (For jieba)
- `<output_file>` is the path to the output prediction file (e.g. `ans.csv`)

```
bash hw6_train.sh <train_x> <train_y> <test_x> <dict.txt.big>
```
- `<train_x>` is the path to `train_x.csv`
- `<train_y>` is the path to `train_y.csv`
<br/>

|          | Public   | Private  |
| :------: | :------: | :------: |
| Accuracy | 0.76280  | 0.76110  |
<br/>


## HW7

### Unsupervised Learning
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw7/)
```
bash pca.sh <images_dir> <input_img> <reconstruct_image>
```
- `<images_dir>` is the directory to the training images (e.g. `Aberdeen/`)
- `<input_img>` is the input image in `<images_dir>` (e.g. `87.jpg`)
- `<reconstruct_image>` is the path to the reconstructed image of `<input_img>` (e.g. `87_reconstruct.jpg`)

```
bash cluster.sh <images_dir> <test_case> <output_file>
```
- `<images_dir>` is the directory to the training images (e.g. `images/`)
- `<test_case>` is the path to `test_case.csv`
- `<output_file>` is the path to the output prediction file (e.g. `ans.csv`)
<br/>

|          | Public   | Private  |
| :------: | :------: | :------: |
| Accuracy | 0.96910  | 0.96892  |
<br/>


## HW8

### Model Compression for Image Sentiment Classification
[Kaggle](https://www.kaggle.com/c/ml2019spring-hw8)
```
mv report.pdf <other_place>
bash hw8_download.sh
>>> Disconnect from the Internet <<<
du hw8 --apparent-size --bytes --max-depth=0
bash hw8_test.sh <testing_data> <output_file>
```
- `<testing_data>` is the path to `test.csv`
- `<output_file>` is the path to the output prediction file (e.g. `ans.csv`)

```
bash hw8_train.sh <training_data>
```
- `<training_data>` is the path to  `train.csv`
<br/>

|            | Public   | Private  |
| :--------: | :------: | :------: |
| Accuracy   | 0.64363  | 0.62803  |
**Model size: 205446 Bytes**
<br/>

