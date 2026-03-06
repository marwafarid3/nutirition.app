import streamlit as st
import pandas as pd
from PIL import Image
import os

# ================================
# LangChain Imports
# ================================
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA

# ================================
# Gemini API Key
# ================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFcvpt-Fs_muflBT96HNZbw4c_9Axa0ik"

# ================================
# LLM
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# ================================
# Nutrition Knowledge Base (RAG)
# ================================
nutrition_data = [
    "Protein sources include chicken, fish, eggs, beans and yogurt.",
    "Healthy carbs include oats, brown rice, sweet potatoes and quinoa.",
    "Healthy fats include avocado, olive oil and nuts.",
    "For weight loss create a calorie deficit.",
    "Drink at least 2 liters of water daily.",
    "Vegetables like broccoli and spinach are low calorie and high nutrients.",
    "Eating protein helps muscle growth.",
    "Avoid excessive sugar and processed foods."
]

docs = [Document(page_content=text) for text in nutrition_data]

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ================================
# Streamlit UI
# ================================
st.title("🥗 Nutrition AI Agent")

# ================================
# User Profile
# ================================
st.sidebar.header("User Profile")

weight = st.sidebar.number_input("Weight (kg)", 40, 200)
height = st.sidebar.number_input("Height (cm)", 140, 210)
age = st.sidebar.number_input("Age", 10, 80)
goal = st.sidebar.selectbox("Goal", ["Lose Weight", "Maintain", "Gain Muscle"])

# ================================
# Calorie Calculation
# ================================
def calculate_calories(weight, height, age):

    bmr = 10*weight + 6.25*height - 5*age + 5

    if goal == "Lose Weight":
        calories = bmr - 400
    elif goal == "Gain Muscle":
        calories = bmr + 300
    else:
        calories = bmr

    return int(calories)

if st.sidebar.button("🔥 Calculate Calories"):
    cals = calculate_calories(weight, height, age)
    st.sidebar.success(f"Daily Calories: {cals}")

# ================================
# Diet Plan Generator
# ================================
st.header("🥗 Generate Diet Plan")

if st.button("Generate Plan"):

    prompt = f"""
    Create a healthy diet plan.

    Weight: {weight}
    Height: {height}
    Age: {age}
    Goal: {goal}

    Include breakfast lunch dinner snacks.
    """

    response = llm.invoke(prompt)

    st.write(response.content)

# ================================
# Food Image Analysis
# ================================
st.header("📷 Analyze Food Image")

uploaded_file = st.file_uploader("Upload food image")

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image)

    prompt = "Describe this food and estimate calories."

    response = llm.invoke([prompt, image])

    st.write(response.content)

# ================================
# Weight Tracking
# ================================
st.header("📊 Weight Tracker")

if "weights" not in st.session_state:
    st.session_state.weights = []

new_weight = st.number_input("Enter today's weight")

if st.button("Add Weight"):
    st.session_state.weights.append(new_weight)

df = pd.DataFrame(st.session_state.weights, columns=["Weight"])

if not df.empty:
    st.line_chart(df)

# ================================
# Chat with AI
# ================================
st.header("💬 Chat with Nutrition Agent")

user_input = st.text_input("Ask about nutrition...")

if user_input:

    response = qa_chain.run(user_input)

    st.write(response)
