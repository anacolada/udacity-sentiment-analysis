# Deploying a sentiment analysis with Amazon SageMaker

This is the first project for the Machine Learning Engineer Nanodegree on Udacity. The project involves an RNN (LSTM+GRU) model which predicts whether a movie review is positive or negative. With regards to deployment, the predictive model endpoint is triggered via a Lambda function, which is called through an AWS API Gateway. The UI is a webpage which allows the end user to input a movie review, and receive an info message indicating the nature of the review.
