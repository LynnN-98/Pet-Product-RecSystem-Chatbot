# app/cli_chat.py

import sys
import os
import threading
import time
from getpass import getpass
from passlib.context import CryptContext
import pickle
from wcwidth import wcswidth
from colorama import Fore, Style, init
import pandas as pd
import textwrap
import warnings
import re

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set environment variables to reduce log output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendations import (
    load_recommendation_system,
    recommend,
    load_hot_products,
    get_product_details,
    content_based_recommendation,
)
from chat_models import get_chat_model_loader
from chat_bot import generate_answer

# Initialize colorama
init(autoreset=True)

# Set up password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Get model loading function
chat_model_loader = get_chat_model_loader()
chat_tokenizer, chat_model = None, None

# Model loading status
model_loaded = False

# Import tqdm for progress bar
from tqdm import tqdm


def load_model_in_background():
    global chat_tokenizer, chat_model, model_loaded
    print(Fore.YELLOW + "Preparing the chatbot... This will just take a moment!" + Style.RESET_ALL)
    for _ in tqdm(range(3), desc="Loading chat model"):
        time.sleep(1)  # Simulate loading process
    chat_tokenizer, chat_model = chat_model_loader()
    model_loaded = True
    print(Fore.GREEN + "Chat model loaded!\n" + Style.RESET_ALL)


def ljust_unicode(s, width, fillchar=" "):
    """Left-align string, considering wide characters"""
    fill_width = width - wcswidth(s)
    if fill_width > 0:
        return s + fillchar * fill_width
    else:
        return s


def load_user_passwords():
    """Load registered user password information"""
    users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_passwords.pkl")
    if os.path.exists(users_file):
        with open(users_file, "rb") as f:
            try:
                users = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                users = {}
    else:
        users = {}
    return users


def save_user_passwords(users):
    """Save user password information to user_passwords.pkl"""
    users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_passwords.pkl")
    with open(users_file, "wb") as f:
        pickle.dump(users, f)


def register(user_id, users):
    """User registration function"""
    print(Fore.GREEN + "=== User Registration ===" + Style.RESET_ALL)

    if user_id in users:
        print(Fore.RED + "This user ID is already registered. Please log in directly." + Style.RESET_ALL)
        return False

    while True:
        password = getpass("Please enter your password: ").strip()
        password_confirm = getpass("Please re-enter your password: ").strip()
        if not password:
            print(Fore.RED + "Password cannot be empty. Please try again." + Style.RESET_ALL)
        elif password != password_confirm:
            print(Fore.RED + "Oops! The passwords didn’t match. Let’s try that again." + Style.RESET_ALL)
        else:
            break
    password_hash = pwd_context.hash(password)

    # Add new user
    users[user_id] = {"password_hash": password_hash}

    # Save user information to user_passwords.pkl
    save_user_passwords(users)
    print(Fore.GREEN + "Registration successful! Please log in with your user ID and password.\n" + Style.RESET_ALL)
    return True


def login():
    """User login function"""
    users = load_user_passwords()
    attempts = 3  # Maximum number of attempts

    while attempts > 0:
        print(Fore.GREEN + "=== User Login ===" + Style.RESET_ALL)
        user_id = input("Please enter your user ID: ").strip()

        if not user_id:
            print(Fore.RED + "User ID cannot be empty." + Style.RESET_ALL)
            attempts -= 1
            if attempts == 0:
                print(Fore.RED + "Multiple invalid user ID inputs. Exiting program." + Style.RESET_ALL)
                return None
            else:
                print(Fore.YELLOW + f"You have {attempts} attempts left.\n" + Style.RESET_ALL)
            continue

        if user_id not in users:
            print(Fore.YELLOW + "This user is not registered." + Style.RESET_ALL)
            choice = input(
                "Would you like to register now? Type 'yes' to register or 'no' to try again with a different User ID: "
            ).strip().lower()
            if choice == "yes":
                if register(user_id, users):
                    # Return to login loop after successful registration
                    continue
                else:
                    attempts -= 1
                    if attempts == 0:
                        print(Fore.RED + "Multiple attempts failed. Exiting program." + Style.RESET_ALL)
                        return None
                    else:
                        print(Fore.YELLOW + f"You have {attempts} attempts left.\n" + Style.RESET_ALL)
                        continue
            elif choice == "no":
                attempts -= 1
                if attempts == 0:
                    print(Fore.RED + "Multiple attempts failed. Exiting program." + Style.RESET_ALL)
                    return None
                else:
                    print(Fore.YELLOW + f"You have {attempts} attempts left.\n" + Style.RESET_ALL)
                    continue
            else:
                print(Fore.YELLOW + "Invalid choice. Please try again.\n" + Style.RESET_ALL)
                continue
        else:
            # User is registered; proceed to password verification
            password_attempts = 3
            while password_attempts > 0:
                password = getpass("Please enter your password: ").strip()
                stored_password_hash = users[user_id]["password_hash"]
                if pwd_context.verify(password, stored_password_hash):
                    print(Fore.GREEN + f"Welcome, {user_id}! Login successful.\n" + Style.RESET_ALL)
                    return user_id
                else:
                    password_attempts -= 1
                    if password_attempts == 0:
                        print(Fore.RED + "Multiple incorrect entries. Login failed." + Style.RESET_ALL)
                        return None
                    else:
                        print(
                            Fore.RED + f"Incorrect password! You have {password_attempts} attempts left.\n" + Style.RESET_ALL
                        )
            return None  # Password verification failed

    print(Fore.RED + "Multiple invalid user ID inputs. Exiting program." + Style.RESET_ALL)
    return None


def display_hot_products(top_5_df):
    """Display hot recommended products in a nice format"""
    print(Fore.GREEN + "=== Hot Recommendations ===" + Style.RESET_ALL)
    if top_5_df.empty:
        print("Sorry, there are no hot products at the moment.")
        return

    for idx, row in top_5_df.iterrows():
        # Define box characters
        box_width = 60
        top_border = "┌" + "─" * box_width + "┐"
        middle_border = "├" + "─" * box_width + "┤"
        bottom_border = "└" + "─" * box_width + "┘"

        print(Fore.YELLOW + top_border + Style.RESET_ALL)
        title = f"│  📦 Product {idx + 1}"
        print(Fore.YELLOW + ljust_unicode(title, box_width + 1) + "│" + Style.RESET_ALL)
        print(Fore.YELLOW + middle_border + Style.RESET_ALL)

        # Get each field's value and handle possible missing values
        asin = f"🏷️ ASIN: {row.get('parent_asin', 'N/A')}"

        description = row.get("description", "")
        if isinstance(description, list):
            description_str = " ".join(map(str, description))
        else:
            description_str = str(description)

        details = row.get("details", "")
        if isinstance(details, dict):
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        else:
            details_str = str(details)

        categories = row.get("categories", "")
        if isinstance(categories, list):
            categories_str = " > ".join(map(str, categories))
        else:
            categories_str = str(categories)

        average_rating = row.get("average_rating", "N/A")
        rating_number = row.get("rating_number", "N/A")
        popularity_score = row.get("popularity_score", "N/A")

        # Use textwrap module to handle long text wrapping
        wrap_width = box_width - 2  # Leave space for border characters
        description_lines = textwrap.wrap(
            f"📝 Description: {description_str}", width=wrap_width, replace_whitespace=False
        )
        details_lines = textwrap.wrap(
            f"🔍 Details: {details_str}", width=wrap_width, replace_whitespace=False
        )
        categories_lines = textwrap.wrap(
            f"📂 Categories: {categories_str}", width=wrap_width, replace_whitespace=False
        )

        # Print product information
        print(Fore.YELLOW + ljust_unicode(f"│ {asin}", box_width + 1) + "│" + Style.RESET_ALL)

        for line in description_lines:
            print(Fore.YELLOW + ljust_unicode(f"│ {line}", box_width + 1) + "│" + Style.RESET_ALL)

        for line in details_lines:
            print(Fore.YELLOW + ljust_unicode(f"│ {line}", box_width + 1) + "│" + Style.RESET_ALL)

        for line in categories_lines:
            print(Fore.YELLOW + ljust_unicode(f"│ {line}", box_width + 1) + "│" + Style.RESET_ALL)

        rating = f"⭐ Average Rating: {average_rating} out of 5 stars"
        print(Fore.YELLOW + ljust_unicode(f"│ {rating}", box_width + 1) + "│" + Style.RESET_ALL)

        rating_num = f"👥 Number of Ratings: {rating_number}"
        print(Fore.YELLOW + ljust_unicode(f"│ {rating_num}", box_width + 1) + "│" + Style.RESET_ALL)

        popularity = f"🔥 Popularity Score: {popularity_score}"
        print(Fore.YELLOW + ljust_unicode(f"│ {popularity}", box_width + 1) + "│" + Style.RESET_ALL)

        print(Fore.YELLOW + bottom_border + Style.RESET_ALL)
        print("\n")


def generate_recommendation_response(
    user_id,
    user_factors,
    item_factors,
    user_id_map,
    item_id_map,
    index,
    loaded_recommendations,
    filtered_data,
    tfidf_vectorizer,
    tfidf_matrix,
):
    """Generate recommendation chat response"""
    recommendations_list = recommend(
        user_id, user_factors, item_factors, user_id_map, item_id_map, index, loaded_recommendations
    )

    user_keywords = None  # Initialize variable

    if recommendations_list:
        # Collaborative filtering has recommendation results
        formatted_recommendations = ""
        for idx, asin in enumerate(recommendations_list[:5], 1):
            formatted_recommendations += f"{idx}. ASIN: {asin}\n"
        response = f"""
Assistant: Certainly! Based on your interests, I recommend the following products:
{formatted_recommendations}
Assistant: Would you like to know more details about any of these products? If so, please enter the corresponding number (e.g., 1).
"""
    else:
        # No recommendation results from collaborative filtering; use content-based filtering
        print(Fore.YELLOW + "Since you're a new user, we need some information to recommend products for you." + Style.RESET_ALL)
        user_keywords = input(Fore.BLUE + "could you tell me what kind of products you're looking for? For example, 'dog toys' or 'cat food': " + Style.RESET_ALL).strip()
        if not user_keywords:
            print(Fore.RED + "No keywords entered; unable to provide recommendations." + Style.RESET_ALL)
            return "Sorry, unable to provide recommendations.", [], None
        content_recommendations = content_based_recommendation(
            user_keywords, tfidf_vectorizer, tfidf_matrix, filtered_data, top_k=5
        )
        if content_recommendations:
            recommendations_list = content_recommendations
            formatted_recommendations = ""
            for idx, asin in enumerate(recommendations_list, 1):
                formatted_recommendations += f"{idx}. ASIN: {asin}\n"
            response = f"""
Assistant: Based on your keywords "{user_keywords}", I recommend the following products:
{formatted_recommendations}
Assistant: Would you like to know more details about any of these products? If so, please enter the corresponding number (e.g., 1).
"""
        else:
            response = "Sorry, unable to find products related to your keywords."
            recommendations_list = []

    return response.strip(), recommendations_list, user_keywords


def main():
    # Start model loading thread
    threading.Thread(target=load_model_in_background, daemon=True).start()

    # Load recommendation system data and display progress bar
    print(Fore.YELLOW + "Loading recommendation system, please wait..." + Style.RESET_ALL)
    for _ in tqdm(range(3), desc="Loading recommendation system"):
        time.sleep(1)  # Simulate loading process

    (
        recommendation_model,
        user_factors,
        item_factors,
        user_id_map,
        item_id_map,
        index,
        loaded_recommendations,
        filtered_data,
        tfidf_vectorizer,
        tfidf_matrix,
    ) = load_recommendation_system()
    print(Fore.GREEN + "Recommendation system loaded!\n" + Style.RESET_ALL)

    # User login
    user_id = login()
    if not user_id:
        sys.exit(1)

    # Load and display hot products
    top_5_df = load_hot_products()
    display_hot_products(top_5_df)

    # Chat loop
    history = []
    print(Fore.GREEN + "=== Let’s get started! Feel free to ask me anything or request product recommendations ===" + Style.RESET_ALL)
    print("Enter 'exit' or 'quit' to end the chat.\n")

    # Define keywords that trigger recommendations
    recommendation_keywords = [
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "interested in",
        "looking for",
        "want to buy",
        "need",
        "show me",
        "find",
        "product",
        "item",
    ]

    while True:
        question = input(Fore.BLUE + "You: " + Style.RESET_ALL).strip()
        if question.lower() in ["exit", "quit"]:
            print(Fore.GREEN + "Thanks for chatting with me! Have a great day!" + Style.RESET_ALL)
            break
        if not question:
            print(Fore.YELLOW + "Please enter a valid question.\n" + Style.RESET_ALL)
            continue

        # Check if the input contains any recommendation keywords
        if any(keyword in question.lower() for keyword in recommendation_keywords):
            # Generate recommendation response and get recommendation list and user keywords
            response, recommendations_list, user_keywords = generate_recommendation_response(
                user_id,
                user_factors,
                item_factors,
                user_id_map,
                item_id_map,
                index,
                loaded_recommendations,
                filtered_data,
                tfidf_vectorizer,
                tfidf_matrix,
            )
            print(Fore.MAGENTA + f"{response}\n" + Style.RESET_ALL)
            # Do not add recommendation reply to history

            # Check if the user wants to inquire about any product details
            need_detail = True
            while need_detail:
                follow_up = input(Fore.BLUE + "You: " + Style.RESET_ALL).strip()
                if follow_up.lower() in ["exit", "quit"]:
                    print(Fore.GREEN + "Chat ended. Goodbye!" + Style.RESET_ALL)
                    sys.exit(0)
                elif follow_up.lower() in ["no", "not needed", "not now", "nope"]:
                    print(Fore.GREEN + "Alright, if you have any other questions, feel free to ask!\n" + Style.RESET_ALL)
                    need_detail = False  # Exit the product detail inquiry loop
                elif follow_up.isdigit():
                    selected_idx = int(follow_up)
                    if 1 <= selected_idx <= len(recommendations_list):
                        selected_asin = recommendations_list[selected_idx - 1]
                        product_details = get_product_details(filtered_data, selected_asin)
                        if product_details:
                            details_response = "Assistant: Here are the details of the product:\n"
                            for key, value in product_details.items():
                                details_response += f"{key}: {value}\n"
                            print(Fore.MAGENTA + f"{details_response}" + Style.RESET_ALL)
                            # Only add user's choice and assistant's detail information to history
                            history.append({"user": follow_up, "assistant": details_response})
                            # Prompt user if they want to know about other products
                            print(Fore.MAGENTA + "Assistant: Would you like to know more details about other products? If so, please enter the corresponding number (e.g., 2). If not, please enter 'no'.\n" + Style.RESET_ALL)
                        else:
                            print(Fore.RED + "Sorry, unable to find details of that product.\n" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + f"Please enter a valid product number (1-{len(recommendations_list)}).\n" + Style.RESET_ALL)
                else:
                    # User input is not a number or 'no'; check if it contains product inquiry intent
                    if any(word in follow_up.lower() for word in ["know", "inquire", "details", "information", "more about"]):
                        # Try to extract number from input
                        match = re.search(r'\d+', follow_up)
                        if match:
                            selected_idx = int(match.group())
                            if 1 <= selected_idx <= len(recommendations_list):
                                selected_asin = recommendations_list[selected_idx - 1]
                                product_details = get_product_details(filtered_data, selected_asin)
                                if product_details:
                                    details_response = "Assistant: Here are the details of the product:\n"
                                    for key, value in product_details.items():
                                        details_response += f"{key}: {value}\n"
                                    print(Fore.MAGENTA + f"{details_response}" + Style.RESET_ALL)
                                    history.append({"user": follow_up, "assistant": details_response})
                                    # Prompt user if they want to know about other products
                                    print(Fore.MAGENTA + "Assistant: Would you like to know more details about other products? If so, please enter the corresponding number (e.g., 2). If not, please enter 'no'.\n" + Style.RESET_ALL)
                                else:
                                    print(Fore.RED + "Sorry, unable to find details of that product.\n" + Style.RESET_ALL)
                            else:
                                print(Fore.RED + f"Please enter a valid product number (1-{len(recommendations_list)}).\n" + Style.RESET_ALL)
                        else:
                            # Unable to extract number; return to Q&A mode
                            print(Fore.GREEN + "Alright, if you have any other questions, feel free to ask!\n" + Style.RESET_ALL)
                            need_detail = False  # Exit the product detail inquiry loop
                    else:
                        # Return to Q&A mode
                        print(Fore.GREEN + "Alright, if you have any other questions, feel free to ask!\n" + Style.RESET_ALL)
                        need_detail = False  # Exit the product detail inquiry loop
            continue  # Return to main loop, waiting for new user input
        else:
            # Handle normal conversation
            if not model_loaded:
                print(Fore.YELLOW + "Chat model is loading, please wait..." + Style.RESET_ALL)
                for _ in tqdm(range(3), desc="Loading chat model"):
                    time.sleep(1)  # Simulate loading process
                while not model_loaded:
                    time.sleep(0.5)
            answer = generate_answer(question, history, chat_tokenizer, chat_model)
            history.append({"user": question, "assistant": answer})
            print(Fore.MAGENTA + f"Assistant: {answer}\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
