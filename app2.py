{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0fb90481",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-02-21 22:20:15.287 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\swati\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "import string\n",
    "import nltk\n",
    "from nltk.tokenize import RegexpTokenizer\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('LRmodel.joblib')\n",
    "\n",
    "# Preprocess the text\n",
    "def preprocess_text(text):\n",
    "    # Convert text to lowercase\n",
    "    text = text.lower()\n",
    "    \n",
    "    # Remove URLs\n",
    "    text = re.sub(r'http\\S+', '', text)\n",
    "    \n",
    "    # Remove punctuation\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
    "    \n",
    "    # Tokenization\n",
    "    tokenizer = RegexpTokenizer(r'\\w+')\n",
    "    tokens = tokenizer.tokenize(text)\n",
    "    \n",
    "    # Remove stopwords\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    filtered_tokens = [word for word in tokens if word not in stop_words]\n",
    "    \n",
    "    # Join tokens back into text\n",
    "    preprocessed_text = ' '.join(filtered_tokens)\n",
    "    \n",
    "    return preprocessed_text\n",
    "\n",
    "# Predict sentiment\n",
    "def predict_sentiment(text, model):\n",
    "    # Preprocess the text\n",
    "    preprocessed_text = preprocess_text(text)\n",
    "    \n",
    "    # Make prediction using the loaded model\n",
    "    vectorized_text = [preprocessed_text]  # TfidfVectorizer not used\n",
    "    prediction = model.predict(vectorized_text)\n",
    "    \n",
    "    return prediction[0]\n",
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
    "        sentiment = predict_sentiment(user_input, model)\n",
    "        \n",
    "        # Display the prediction result\n",
    "        if sentiment == 1:\n",
    "            st.success('Positive Sentiment')\n",
    "        else:\n",
    "            st.error('Negative Sentiment')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1705ea7",
   "metadata": {},
   "outputs": [],
   "source": []
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
