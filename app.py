import re
import pdfplumber
from docx import Document
import nltk
import streamlit as st
import requests

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------- TEXT PREPROCESS -----------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\+', 'plus', text)
    text = re.sub(r'#', 'sharp', text)
    text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ----------------- READ FILES -----------------
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file):
    return file.read().decode('utf-8')

def read_file(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return read_pdf(file)
    elif ext == 'docx':
        return read_docx(file)
    elif ext == 'txt':
        return read_txt(file)
    else:
        return ""

# ----------------- EXTRACT SKILLS -----------------
def extract_skills_from_jd(text, top_n=20):
    words = text.split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    jd_skills = [word for word, count in sorted_words[:top_n]]
    return jd_skills

# ----------------- JD COVERAGE SCORE -----------------
def jd_coverage_score(jd_keywords, resume_text):
    resume_words = set(resume_text.split())
    matched_keywords = [word for word in jd_keywords if word in resume_words]
    missing_keywords = [word for word in jd_keywords if word not in resume_words]
    score = len(matched_keywords) / len(jd_keywords)
    return score, matched_keywords, missing_keywords

# ----------------- Hugging Face AI Suggestions -----------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HF_API_KEY = "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"  # replace with your key

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def hf_suggestions(missing_skills):
    prompt = f"""
    The candidate's resume is missing these skills: {', '.join(missing_skills)}.
    Suggest ways to improve the resume to match the job description better.
    """
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    try:
        return response.json()[0]['generated_text']
    except:
        return "AI suggestion not available at the moment."

# ----------------- STREAMLIT APP -----------------
st.title("📄 Resume JD Match Checker with AI Feedback (Hugging Face API)")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=['pdf','docx','txt'])
jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=['pdf','docx','txt'])

if resume_file and jd_file:
    resume_text = preprocess_text(read_file(resume_file))
    jd_text = preprocess_text(read_file(jd_file))
    
    jd_keywords = extract_skills_from_jd(jd_text, top_n=20)
    st.write("🔹 Extracted JD keywords:", jd_keywords)
    
    score, matched, missing = jd_coverage_score(jd_keywords, resume_text)
    st.success(f"✅ JD Coverage Score: {round(score*100, 2)}%")
    
    if missing:
        st.warning(f"⚠️ Missing keywords in resume: {', '.join(missing)}")
        st.info("💡 AI Suggestions to improve your resume:")
        suggestions = hf_suggestions(missing)
        st.write(suggestions)
