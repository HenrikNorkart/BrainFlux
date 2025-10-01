from dotenv import load_dotenv

load_dotenv()

try:
    load_dotenv(".dev.env", override=True)
except Exception:
    print("No .dev.env file found, continuing...")
