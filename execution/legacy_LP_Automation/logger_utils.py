import json
from datetime import datetime
from colorama import Fore, Style, init

# Initialize color support for PowerShell
init(autoreset=True)

# ------------------------------------------------
#  Format a neat cycle summary (with colors)
# ------------------------------------------------
def format_cycle_summary(data):
    """Creates a human-readable, color-coded summary for logs."""
    profit = data.get("cycle_profit_usd", 0)
    pnl = data.get("pnl_usd", 0)
    profit_color = Fore.GREEN if profit >= 0 else Fore.RED
    pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED

    return (
        f"\n==================================================\n"
        f"📘  Cycle Summary\n"
        f"==================================================\n"
        f"🕒 Withdraw: {data.get('timestamp_withdraw', 'N/A')}\n"
        f"🕓  Created:  {data.get('timestamp_create', 'N/A')}\n"
        f"🗓️  Logged at: {data.get('logged_at', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))}\n\n"
        f"💵 Fees Collected:     +${data.get('fees_collected_usd', 0):.2f}\n"
        f"{pnl_color}📊 Rebalance PnL:      {pnl:+.2f} USD{Style.RESET_ALL}\n"
        f"{profit_color}✅ Total Cycle Profit:  {profit:+.2f} USD{Style.RESET_ALL}\n"
        f"🎯 Range Width:        {data.get('range_width', 'N/A')} ticks\n"
        f"==================================================\n"
    )

# ------------------------------------------------
#  Save structured and formatted logs
# ------------------------------------------------
def save_cycle_data(data, append=True):
    """Writes last cycle + optionally appends formatted summary to history."""
    # Update metadata
    data["logged_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    data["updated_at"] = data["logged_at"]

    # Save structured version
    with open("last_cycle_data.json", "w") as f:
        json.dump(data, f, indent=2)

    if append:
        # Append both JSON and formatted summary
        with open("cycle_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            f.write(format_cycle_summary(data))
            f.write("\n")

    # Also print a concise version to PowerShell
    print(format_cycle_summary(data))
    print(f"💾 Recorded PnL data → {data}")
