# WSGI configuration for PythonAnywhere
import sys
import os

# Add your project directory to the Python path
project_home = '/home/yourusername/cutline-api'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import your Flask app
from app import app as application

if __name__ == "__main__":
    application.run()
