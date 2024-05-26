import re
import json

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

    # Store User Credentials
    def store_user_details(self,
                           first_name,
                           last_name,
                           email,
                           username,
                           password):
        
        # Store user details
        self.users[username] = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
        }

        with open("user_data.json", 'w') as file:
            json.dump(self.users, file, indent=4)

    # Log In Function
    def log_in(self, username, password):
        # Check if username exists and password matches
        if username in self.users and self.users[username]['password'] == password:
            return True
        else:
            return False