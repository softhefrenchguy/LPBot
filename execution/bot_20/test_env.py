import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

print("ðŸ“ Current working directory:", os.getcwd())
loaded = load_dotenv(".env")
print("âœ… load_dotenv() returned:", loaded)

wallet = os.getenv("WALLET_ADDRESS")
key = os.getenv("PRIVATE_KEY")

print("ðŸ”¹ WALLET_ADDRESS =", wallet)
print("ðŸ”¹ PRIVATE_KEY =", key[:10] + "..." if key else None)
