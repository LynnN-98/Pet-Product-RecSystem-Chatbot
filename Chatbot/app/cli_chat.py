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