import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import requests
import json
import os

# --- Config ---
OUTPUT_DIR = "abis"
ABI_URLS = {
    "NonfungiblePositionManager.json":
        "https://raw.githubusercontent.com/Uniswap/v3-periphery/main/artifacts/contracts/NonfungiblePositionManager.sol/NonfungiblePositionManager.json",
    "UniswapV3Pool.json":
        "https://raw.githubusercontent.com/Uniswap/v3-core/main/artifacts/contracts/UniswapV3Pool.sol/UniswapV3Pool.json"
}

# --- Function to download and save ---
def download_abi(filename, url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        abi = data.get("abi")
        if abi is None:
            print(f"â— No 'abi' field found in {filename} at {url}")
            return False
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(abi, f, indent=2)
        print(f"âœ… Saved {filename} to {path}")
        return True
    except Exception as e:
        print(f"â— Failed to download {filename} from {url}: {e}")
        return False

# --- Main execution ---
if __name__ == "__main__":
    for fname, url in ABI_URLS.items():
        print(f"Downloading {fname} â€¦")
        download_abi(fname, url)

    print("\nDone. Check the `abis/` folder for the files.")
