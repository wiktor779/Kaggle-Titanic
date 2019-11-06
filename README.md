# Kaggle-Titanic (top 9% solution, 80% accuracy score)

## Introduction
The goal of this project was to use deep learning to create a model that predicts which passengers 
survived the Titanic shipwreck. 

My solution scored a little over <b>80%</b> accuracy, which classifies it among top <b>9%</b> solutions.

![MY_PREDICTION](https://user-images.githubusercontent.com/20689930/68073526-749c5b00-fd91-11e9-947f-adeb41a4f29d.png)

## Input data

Input data consists of 1309 samples, most of them had the following features: 
- survived - Did the passanger survive? 0 = No, 1 = Yes
- pclass - Ticket class. 1 = 1st, 2 = 2nd, 3 = 3rd
- name - Passenger name
- sex - Passenger sex
- age - Passenger age
- Sibsp - Number of siblings / spouses aboard the Titanic	
- Parch - NUmber of parents / children aboard the Titanic
- Ticket - Ticket number
- Fare - Passenger fare
- Cabin - Cabin number
- Embarked - Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Data was split into two parts: first for training, second for testing model. I trained neural network on training data 
to predict as much survivors as possible. As shown in the table below there is NaN value for testing data in "Survived"
column, which i have to predict.

![image](https://user-images.githubusercontent.com/20689930/68071199-7f96c180-fd78-11e9-8097-3b0a86cd267e.png)

## About project
In Source folder you can find 3 files:
- [proprocessing.ipynb](https://github.com/wiktor779/Kaggle-Titanic/blob/master/Source/proprocessing.ipynb) - loading 
data and some preprocessing like One hot encoding and data normalization.
- [age_prediction.ipynb](https://github.com/wiktor779/Kaggle-Titanic/blob/master/Source/age_prediction.ipynb) - simple
model for predicting age
- [survived_prediction.ipynb](https://github.com/wiktor779/Kaggle-Titanic/blob/master/Source/age_prediction.ipynb) - main
model used to predict if passenger survived or not

## Getting started
Because github sometimes has problems with showing .ipynb files you can use <b>Google colab</b> to view files.
Go into [this](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) site. Choose "GitHub" from orange
list, paste "https://github.com/wiktor779/Kaggle-Titanic" and click search button.

## Preprocessing

### First look at data
Train data consist of 891 sample and train data from 418. I concatenated this data and counted empty values for each
feature. Because up to a thousand values are missing in "Cabin" I decided to drop this feature. There are also 263 
values missing for "Age". Because I didn't want to fill empty values with mean I decided to create a simple neural 
network to predict missing age which I will write about later.
![emptyValues](https://user-images.githubusercontent.com/20689930/68233111-ceee2380-fffe-11e9-8a1d-852aa18d7a86.png)

### One hot encoding
Because features like: "Name", "Sex", "Ticket", "Embarked" are in the form of a string and neural network can deal only 
with numbers i had to change this. I started from counting occurrences of each word in "Name" and "Ticket" columns in 
order to create new columns for the most common words. I received the following results.

![countWords](https://user-images.githubusercontent.com/20689930/68234935-994b3980-0002-11ea-9b75-56e6f071bbd0.png)

Next i performed one hot encoding for all categorical columns. I received the following results.

![oneHotEncoding](https://user-images.githubusercontent.com/20689930/68295347-81bb9180-0092-11ea-8a1c-e4645b60a52e.png)

As you can see all categorical columns was changed into binary values. 

### Data normalization

Last step in data preprocessing was to normalize the data. It is important to remember to apply the exact same scaling 
for test data as for training data. I used MinMaxScaler from sklearn to normalize data to 0-1 range.

![normalizeDate](https://user-images.githubusercontent.com/20689930/68296431-e4ae2800-0094-11ea-84e0-1feed1ac8b9d.png)





## Age prediction

As I mentioned before I had to deal with missing age values. I created simple neural network with two hidden layers
to predict this values. I used RMSprop optimization alhoritm and Mean Square Error loss function. 

![age_prediction](https://user-images.githubusercontent.com/20689930/68297168-85511780-0096-11ea-867c-d176838ca5bc.png)

As you can see on histogram below predicted values has similar trends to training values.

![image](https://user-images.githubusercontent.com/20689930/68298169-fdb8d800-0098-11ea-8800-edc3e78c782b.png)

## Survivors prediction

### K-fold validation
To evaluate my network while I was keep adjusting its parameters (such as the number of epochs) I split the data
into a training set and a validation set. But because I have so few data points, the validation set would end up being 
very small. As a consequence, the validation scores could change a lot depending on which data points I chose to use for
validation and which I chose for training. To prevent this i used K-fold cross-validation. It consists of splitting
the available data into K partitions (in this case K = 7), instantiating K identical models, and training each one on
K-1 partitions while evaluating on the remaining partition.

### Creating model and hyperparameter tuning
As in the age prediction I created a sequential model. After many tries I decided that model with Dense layers
separated by Dropout layers works best. I started with 64 neurons in first hidden layers and reduced neurons number 
in two subsequent layers. Between these layers I added Dropout layer to prevent overfeeding.

![image](https://user-images.githubusercontent.com/20689930/68333148-4f348780-00d8-11ea-9613-5305ea3cf336.png)

### Model training
Below you can see training and validation loss/accuracy. For final training I decided to train network for 30 epochs 
because after this time model seems to stop improving.

![image](https://user-images.githubusercontent.com/20689930/68335040-0c74ae80-00dc-11ea-95b8-18acd434edc5.png)

##Summary
My final score is 0.80382 which classify my solution among top 9%. It is significant improvement compared to primitive
solution (all women survived, all men died) which score 0.76555.

## Built With

* [Keras](https://keras.io/) - creating and training models
* [Pandas](https://pandas.pydata.org/) - data preprocessing 
* [Numpy](https://numpy.org/) - concatenating arrays
* [Sklearn](https://scikit-learn.org/stable/) -normalization data
* [Matplotlib](https://matplotlib.org/) - visualizing results

## Authors

* [wiktor779](https://github.com/wiktor779)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

