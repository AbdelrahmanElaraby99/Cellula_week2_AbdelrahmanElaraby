# 🧪 NLP Internship - Task 2: Toxic Content Classification Website using Streamlit.

Welcome to the **Toxic Content Classification Project** at **Cellula Technologies**! This project focuses on building an NLP application that classifies text (and image captions) as toxic or non-toxic, while exploring model optimization techniques such as **quantization** for large models.  

---

## 📂 Project Structure

```
Cellula_week2_AbdelrahmanElaraby
¦   README.md
¦   
+---TASK0_Research
¦       Quantization.pdf
¦       Quantization_tests.ipynb
¦       
+---TASK1_Code
    ¦   classifier.py
    ¦   db_creation.py
    ¦   imagecaption.py
    ¦   main.py
    ¦   requirements.txt
    ¦   
    +---.streamlit
    ¦       config.toml
    ¦       
    ¦       
    +---helping_files
    ¦       best_model.weights.h5
    ¦       cellula.jpg
    ¦       label_encoder.pkl
    ¦       tokenizer.pkl
    ¦       website.db

```

---

## 🔍 Task Overview

### Task 0: Research Part
Modern NLP models like **BERT** or **LLaMA** are extremely large and resource-intensive. To handle this, we explore **quantization** as a technique to reduce model size and improve inference efficiency without sacrificing much accuracy.

- ✅ Discuss quantization techniques  
- ✅ Provide coding examples  
- ✅ Use LaTeX for formulas (bonus points)  
- ✅ Include graphs/images for better understanding (bonus points)  

---

### Task 1: Toxic Content Classification Project

This task involves building a full **text and image caption classification application** using **Streamlit**.  

#### 1️⃣ File Format
- Only `.py` files are allowed.
- Jupyter Notebooks (`.ipynb`) are **not permitted**.

#### 2️⃣ Modular Code Structure
- Implement the **image captioning function** in `imagecaption.py`.  
- Import this module into the main script for use.  

#### 3️⃣ Model Selection
- **Image Captioning:** BLIP-1, BLIP-2 (Hugging Face)  
- **Text Classification:**  
  - LSTM  
  - LLaMA Guard  
  - Fine-tuned DistilBERT with LoRA  
  - Fine-tuned ALBERT with LoRA  

#### 4️⃣ Database Management
- Use a **CSV file** as the database. (or use sqlite3 if you want) 
- The CSV should automatically update whenever a user submits:
  - Text input  
  - Generated image caption  
- Store in the database:
  - User input or generated caption  
  - Corresponding classification result  

#### 5️⃣ Application Framework
- Use **Streamlit** to develop the interactive application.

#### 6️⃣ Database Viewing Option
- Include a feature to view all stored inputs and their classifications at any time.  
- This ensures transparency and easy debugging.  

