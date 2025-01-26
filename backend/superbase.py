import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPERBASE_URL = os.getenv("SUPERBASE_URL")
SUPERBASE_KEY = os.getenv("SUPERBASE_ANON_KEY")

create_client(SUPERBASE_URL, SUPERBASE_KEY)