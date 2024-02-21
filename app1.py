{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5b3294d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('LRmodel.joblib')\n",
    "\n",
    "# Function to predict sentiment\n",
    "def predict_sentiment(text):\n",
    "    # Preprocess the text (you'll need to define preprocessing steps)\n",
    "    # Vectorize the preprocessed text using the same TfidfVectorizer used during training\n",
    "    # Predict the sentiment using the loaded model\n",
    "    # Return the sentiment prediction\n",
    "    pass\n",
    "\n",
    "# Streamlit app\n",
    "def main():\n",
    "    st.title('Twitter Sentiment Analysis')\n",
    "    st.write('Enter your tweet below:')\n",
    "    \n",
    "    # Text input for user to enter tweet\n",
    "    user_input = st.text_input('Input Tweet:')\n",
    "    \n",
    "    if st.button('Predict'):\n",
    "        # Predict sentiment when button is clicked\n",
    "        sentiment = predict_sentiment(user_input)\n",
    "        \n",
    "        # Display the prediction result\n",
    "        if sentiment == 1:\n",
    "            st.success('Positive Sentiment')\n",
    "        else:\n",
    "            st.error('Negative Sentiment')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
