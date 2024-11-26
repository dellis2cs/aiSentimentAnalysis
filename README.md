# Product Sentiment Analysis

## Description

This project is a Logistic Regression model trained and fine tuned using pscikit-learn and python, the frontend is a react server connected to a backend flask server
## Features
- Input a product review and the ML model will determine whether the review is positive or negative

## Table of Contents
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Running the Application](#usage)

## Technologies Used

- **React** - For building the user interface.
- **Flask** - For handling routes and connecting to the frontend.
- **Python** - For training the ai prediction model on the data
- **CSS3** - For styling the layout.

## Getting Started

### Prerequisites

- **Node.js**, **npm**, and **python** installed on your system.

### Installation

- To get started, clone the repository and install dependencies
  
   ```bash
   git clone git@github.com:dellis2cs/HaiSentimentAnalysis.git
   ```
   ```bash
   cd aiSentimentAnalysis
   ```
   ```bash
   npm install
   ```
   ```bash
   pip install flask
   pip install pandas
   pip install scikit-learn
   pip install flask_cors
   pip install nltk
   pip install pickle
   ```

## Usage

Run the application
```bash
cd frontend
npm run dev
   ```
```bash
cd backend
python3 app.py
   ```

## Ways to improve the model
- The main issue I ran into when training this model was that when encountering phrases, not individual words the AI struggled
- For example if I said "this product is not fun", it would see the word fun and mark it positive
- I have trained and fine tuned to keep these discrepancies to a minimum, however you could transition to using a more power and context aware model such as bert
