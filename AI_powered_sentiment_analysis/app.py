import sys
import streamlit as st

try:
    import joblib
    import pandas as pd
except ImportError:
    st.error(f"Wrong Python is running this app: {sys.executable}")
    st.stop()

# load the saved pipeline once, not on every click
model = st.cache_resource(joblib.load)("sentiment_pipeline.pkl")

st.title("Sentiment Analysis")
st.write("Type a customer review and the model will predict if it is Positive, Negative or Neutral.")

review = st.text_area("Enter a review")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        prediction = model.predict([review])[0]
        probs = pd.Series(model.predict_proba([review])[0], index=model.classes_)

        st.success(f"Predicted sentiment: {prediction}")
        st.write(f"Confidence: {probs.max():.1%}")
        st.bar_chart(probs)