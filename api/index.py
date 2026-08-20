"""
Vercel Serverless Function Entry Point for FastAPI Backend.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.main import app

# Export app for Vercel Serverless Runtime
handler = app
