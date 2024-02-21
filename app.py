{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "874f8002",
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
    "    prediction = model.predict([text])\n",
    "    return prediction[0]\n",
    "\n",
    "# Streamlit app\n",
    "def main():\n",
    "    st.title('Sentiment Analysis')\n",
    "    st.write('Enter your text below to analyze sentiment:')\n",
    "    \n",
    "    # Input text area\n",
    "    text_input = st.text_area('Input Text')\n",
    "    \n",
    "    # Button to trigger sentiment analysis\n",
    "    if st.button('Analyze'):\n",
    "        if text_input:\n",
    "            # Perform sentiment analysis\n",
    "            prediction = predict_sentiment(text_input)\n",
    "            st.write('Sentiment:', prediction)\n",
    "        else:\n",
    "            st.write('Please enter some text to analyze.')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3a1279ee",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b3d66a6d",
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
