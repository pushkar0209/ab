import os
import sys

# Append parent directory to sys.path so we can import dashboard and src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import app

# This allows Vercel to find the WSGI application
