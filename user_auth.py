import re
import json
import pandas as pd

class UserAuthentication:
    def __init__(self):
        # Load existing user data from file, if available
        try:
            with open('user_data.json', 'r') as file:
                self.users = json.load(file)
        except FileNotFoundError:
            self.users = {}
    
    def sign_up(self,
                first_name,
                last_name,
                email,
                username,
                password):
        
        # Promt user for details
        # st.title("Sign Up")
        # first_name = st.text_input("First Name")
        # last_name = st.text_input("Last Name")
        # email = st.text_input("Email")
        # username = st.text_input("Username")
        # password = st.text_input("Password", type='password')
        
        # Check if username already exists
        if username in self.users:
            return False
        
        # Check email requirements
        if not self.check_email_requirements(email):
            return False
        
        # Check password requirements
        if not self.check_password_requirements(password):
            return False
        


        self.store_user_details(first_name,
                                last_name,
                                email,
                                username,
                                password)


        return True
    
    # Check password requirements function
    def check_password_requirements(self, password):
        # Check password requirements
        if len(password) < 8:
            return False
        if not re.search("[a-z]", password):
            return False
        if not re.search("[A-Z]", password):
            return False
        if not re.search("[0-9]", password):
            return False
        if not re.search("[!@#$%^&]", password):
            return False
        return True
    
    # Check email requirements function
    def check_email_requirements(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if re.match(pattern, email):
            return True
        else:
            return False

    def store_user_details(self, first_name, last_name, email, username, password):
        self.users[username] = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
            'searches': []  # Initialize an empty list to store user searches
        }
        self.save_user_data()

    def save_user_data(self):
        with open("user_data.json", 'w') as file:
            json.dump(self.users, file, indent=4)

    def log_in(self, username, password):
        if username in self.users and self.users[username]['password'] == password:
            return True
        return False

    def save_search(self, username, search_data):
        if username in self.users:
            self.users[username]['searches'].append(search_data)
            self.save_user_data()

    def get_user_searches(self, username):
        if username in self.users:
            return self.users[username].get('searches', [])
        return []

    def save_result(self, username, dataframe):
        if username in self.users:
            search_data = dataframe.to_dict()
            self.save_search(username, search_data)

    def get_user_results(self, username):
        if username in self.users:
            searches = self.users[username].get('searches', [])
            return [pd.DataFrame(search) for search in searches]
        return []