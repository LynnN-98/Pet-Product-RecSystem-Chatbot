# Chatbot: AI-Powered Chat & Recommendation System

## Overview

This project is an AI-powered chatbot integrated with a **fine-tuned GPT-2 model** for natural language conversations and a **hybrid recommendation system** combining **collaborative filtering (SVD++)** and **content-based filtering (TF-IDF & FAISS)**.

### Key Features

- **User Authentication:** Secure user login and registration with hashed passwords.
- **Conversational AI:** Engage in intelligent conversations with a fine-tuned GPT-2 chatbot.
- **Personalized Recommendations:** Receive tailored product recommendations based on your history.
- **Trending Products:** View the most popular recommended products.
- **Product Details Lookup:** Retrieve detailed information about recommended products.
![Pet流程图](https://github.com/user-attachments/assets/82708d05-958f-4de7-adbf-04f90100a098)

---

## Project Structure

```
Chatbot/
├── Dockerfile
├── app
│   ├── __init__.py
│   ├── chat_bot.py
│   ├── chat_models.py
│   ├── cli_chat.py  # Main chat program
│   ├── recommendations.py
│   └── user_passwords.pkl  # Stores hashed user passwords
├── models_cli
│   └── gpt2  # Fine-tuned GPT-2 model
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── merges.txt
│       ├── tokenizer_config.json
│       └── vocab.json
├── recommendations
│   ├── SVD++_best_model.pkl  # Collaborative filtering model
│   ├── item_index.faiss  # FAISS index for fast product search
│   ├── tfidf_vectorizer.pkl  # Content filtering vectorizer
│   └── top_5.csv  # Trending products data
├── requirements.txt
└── test.ipynb  # Jupyter notebook for testing
```

---

## Getting Started

You can run this chatbot **locally** or **inside a Docker container**.

### **Option 1: Run Locally**

#### 1️⃣ Set Up the Environment

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate  # For Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot:
   ```bash
   python app/cli_chat.py
   ```

---

### **Option 2: Run with Docker**

#### 1️⃣ Pull the Docker Image

```bash
docker pull lynn7777/llm_gpt-2:latest
```

#### 2️⃣ Run the Container

```bash
docker run -it --rm lynn7777/llm_gpt-2:latest
```

---

## **How It Works**

### **1️⃣ User Authentication**

- First-time users **register** with a username and password.
- Returning users **log in** securely with hashed passwords stored in `user_passwords.pkl`.
- The password is encrypted using `passlib` before being stored.

### **2️⃣ Chatbot Conversations**

- Users interact with the chatbot via text input.
- The chatbot processes inputs using **GPT-2**, generating context-aware responses.
- If a user requests recommendations, the chatbot triggers the recommendation system.

### **3️⃣ Personalized Recommendations**

- If a returning user, the **SVD++ collaborative filtering model** provides personalized product recommendations based on past interactions.
- If a new user, the system prompts them to enter **keywords of interest**, and then it suggests products using **content-based filtering (TF-IDF & FAISS)**.
- If no preference data is available, the chatbot shows trending products.

### **4️⃣ Viewing Product Details**

- Users can **input a product ID** to get detailed information about a specific product.
- The system fetches ASIN, description, rating, and other details from preprocessed files.

---

## User Guide

### 1. Start the Program

- **Run Locally:**

  ```bash
  python cli_chat.py
  ```

- **Run with Docker:**

  ```bash
  docker run -it --rm lynn7777/llm_gpt-2:latest
  ```

### 2. Login or Register

**First-time users need to register by entering a username and password as prompted.**

**Note:** The **USER_ID** is used as the username, and the default password for all users is `password123`. This information is stored in `user_passwords.pkl`. If the username does not exist, the system will register the user and save their credentials.

```
Loaded user-password mappings (first 5 entries):
AE22236AFRRSMQIKGG7TPTB75QEA: password123
AE222MW56PH6JXPIB6XSAMCBTLNQ: password123
AE222N3VUKMF3GO6D4LHTELE7UWA: password123
AE2244ILMBLRPTIN7VW7YDKRI2YA: password123
AE226BJM6RTWIVV6UJKZAVQPBKXA: password123
```

- **After starting the program, the system prompts for a username:**

  ```bash
  Please enter your username:
  ```

- **New User Registration:**

  - If the username does not exist, the system prompts:

    ```bash
    This user is not registered.
    Would you like to register now? Enter 'yes' to register or 'no' to re-enter a username:
    ```

  - Enter `yes` to register. The system will then prompt for a password:

    ```bash
    Please enter your password:
    Please re-enter your password:
    ```

  - **Password Requirements:**

    - The password cannot be empty and must match on both entries.
    - Passwords are securely stored using a hashing algorithm.

  - Upon successful registration, the system prompts the user to log in.

- **Login for Existing Users:**

  - Enter the registered username, and the system prompts for a password:

    ```bash
    Please enter your password:
    ```

  - **Password Verification:**

    - If correct, login is successful.
    - If incorrect, users have **three attempts** before being locked out.

### 3. View Trending Recommendations

**After logging in, the system displays the current most popular products.**

- The system automatically loads and presents **the top 5 trending products** in a list format:

  ```
  === Trending Recommendations ===
  ┌──────────────────────────────────────────────┐
  │  📦 Product 1                                │
  ├──────────────────────────────────────────────┤
  │ 🏷️ ASIN: B00XXXXXXX                         │
  │ 📝 Description: High-quality pet food...     │
  │ ⭐ Average Rating: 4.5 ★                      │
  │ 👥 Number of Reviews: 250                    │
  │ 🔥 Popularity Score: 95                      │
  └──────────────────────────────────────────────┘
  ...
  ```

- **Information displayed includes:**

  - Product number
  - ASIN (Amazon Standard Identification Number)
  - Product description
  - Average rating
  - Number of reviews
  - Popularity score

### 4. Start a Chat

**Enter a question or request to interact with the chatbot.**

- **Example Input:**

  ```bash
  You: Hello, what can you do?
  ```

- **Bot Response:**

  ```bash
  Assistant: Hello! I am your AI assistant. I can provide product recommendations, answer questions, and more.
  ```

- **Interaction Features:**

  - Users can freely ask questions, and the chatbot will generate responses.
  - Supports casual conversation, inquiries, and information retrieval.

### 5. Get Product Recommendations

**When users request product recommendations, the chatbot generates a personalized list.**

- **Triggering the Recommendation Feature:**

  - The chatbot recognizes certain keywords related to recommendations:

    - "recommend"
    - "suggest"
    - "interested in"
    - "looking for"
    - "want to buy"
    - "show me"
    - "find"
    - "product"

  - **Example:**

    ```bash
    You: Can you recommend some pet products?
    ```

- **Generating Recommendations:**

  - **For existing users:**

    - The system provides **personalized recommendations** based on historical data using collaborative filtering.

  - **For new users:**

    - The system prompts the user to enter relevant **keywords** for content-based recommendations:

      ```bash
      Since you are a new user, we need some information to provide recommendations.
      Please enter product-related keywords:
      ```

    - **Example Input:**

      ```bash
      You: dog food
      ```

- **Viewing Recommendations:**

  - The system displays the recommended products along with ASINs:

    ```bash
    Assistant: Based on your interest, here are some recommendations:
    1. ASIN: B00XXXXXXX
    2. ASIN: B00YYYYYYY
    3. ASIN: B00ZZZZZZZ
    ...
    Would you like to see more details? If so, enter the corresponding number (e.g., 1).
    ```

### 6. Check Product Details

**Enter the corresponding product number to get more details.**

- **Fetching Product Details:**

  - **Enter the product number:**

    ```bash
    You: 1
    ```

  - **System Response:**

    ```bash
    Assistant: Here are the details of this product:
    ASIN: B00XXXXXXX
    Description: A high-quality pet product, suitable for all breeds...
    Detailed Information: Brand: XX, Size: XX, Weight: XX...
    Category: Pet Supplies > Dog Products
    Average Rating: 4.5 ★
    Number of Reviews: 250
    Popularity Score: 95
    ```

- **Continue Searching:**

  - The system will ask if the user wants more product details:

    ```bash
    Assistant: Would you like to see details for another product? If yes, enter the number; if not, type 'no'.
    ```

  - **Input 'no' to return to the chat:**

    ```bash
    You: no
    ```

### 7. Exit the Program

**At any time, type `exit` or `quit` to leave the chat.**

- **Example:**

  ```bash
  You: exit
  Assistant: Chat ended. Goodbye!
  ```

- The system saves chat history and user preferences for future interactions.

---

## Notes

- **Data File Integrity:**

  - If using Docker, all required data and model files are included in the image.
  - If running from source, ensure all necessary files exist in `recommendations/` and `models_cli/gpt2/`.

- **Model Loading Time:**

  - The chatbot may take some time to load the **GPT-2 model** initially.

- **Error Handling:**

  - If an `OSError` appears, ignore it—it does not affect functionality.

- **Security:**

  - Passwords are securely hashed before storage.

- **Input Requirements:**

  - Ensure correct product numbers and structured queries for better responses.

---

## **Deployment Plans**

Currently, the project runs locally and via Docker. Deployment via **FastAPI** for REST API functionality and **AWS SageMaker Inference** is under development.

### **Future Deployment Steps**

- Implement **FastAPI** for serving model responses as an API.
- Deploy the API using **Docker**.
- Push the container to **AWS ECR**.
- Deploy as a **SageMaker Inference Endpoint** for cloud-based access.

---

## **Final Notes**

- For **local testing**, use `python app/cli_chat.py`.
- For **Docker access**, use `docker run -p 8000:8000 chatbot-fastapi` (coming soon).
- For **cloud scalability**, future deployment will support **AWS SageMaker & ECS**.

🚀 **Stay tuned for upcoming deployment updates!**
