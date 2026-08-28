import re
import json
from datetime import datetime, timedelta, date

INPUT_FILE = "discord_cycles.txt"
OUTPUT_FILE = "cycle_history.jsonl"

# We know from context:
# - All cycles are in 2025
# - "Yesterday at HH:MM" is 2025-12-02
# - Bare "HH:MM" at the end are 2025-12-03
YESTERDAY_DATE = date(2025, 12, 2)
TODAY_DATE = date(2025, 12, 3)


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_header_datetime(lines, idx):
    """
    Look upwards from line idx to find a timestamp line.
    Supported formats:
      - " — 26/11/2025 16:41"
      - " — Yesterday at 08:31"
      - " — 02:12"  (assumed today)
    """
    for j in range(idx, -1, -1):
        text = lines[j].strip()
        if not text:
            continue

        # Explicit date: 28/11/2025 14:01
        m = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})", text)
        if m:
            d_str, t_str = m.groups()
            return datetime.strptime(d_str + " " + t_str, "%d/%m/%Y %H:%M")

        # "Yesterday at HH:MM"
        m = re.search(r"Yesterday at\s+(\d{2}:\d{2})", text)
        if m:
            t_str = m.group(1)
            t = datetime.strptime(t_str, "%H:%M").time()
            return datetime.combine(YESTERDAY_DATE, t)

        # Bare "HH:MM" with no date → assume today
        m = re.search(r"(\d{2}:\d{2})$", text)
        if m and "Yesterday" not in text and "/" not in text:
            t_str = m.group(1)
            t = datetime.strptime(t_str, "%H:%M").time()
            return datetime.combine(TODAY_DATE, t)

    return None


def parse_cycles(lines):
    cycles = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        m_cycle = re.search(r"LP Cycle #(\d+)", line)
        if not m_cycle:
            i += 1
            continue

        cycle_number = int(m_cycle.group(1))
        cycle = {
            "cycle_number": cycle_number,
        }

        # Get withdraw timestamp from header above
        ts_exit = parse_header_datetime(lines, i)
        if ts_exit:
            cycle["timestamp_exit"] = ts_exit.isoformat() + "Z"
        else:
            cycle["timestamp_exit"] = None

        # Scan forward until next "LP Cycle #" or end
        j = i + 1
        duration_minutes = None
        wallet_pnl = None
        lp_pnl = None
        value_in = None
        value_out = None
        fees_sent = None
        exit_reason = None

        while j < n:
            text = lines[j].strip()

            # Stop at next cycle
            if "LP Cycle #" in text and j != i:
                break

            # Duration: "Duration: 1118.76 min"
            m_dur = re.search(r"Duration:\s*([\d\.]+)\s*min", text)
            if m_dur:
                try:
                    duration_minutes = float(m_dur.group(1))
                except ValueError:
                    pass

            # Wallet PnL after fees/gas
            if "Wallet PnL after fees" in text:
                m_p = re.search(r"Wallet PnL after fees:\s*([+\-]?\d+[\d\.]*)\s*USDC", text)
                if m_p:
                    wallet_pnl = float(m_p.group(1))
            elif "PnL (wallet):" in text:
                m_p = re.search(r"PnL \(wallet\):\s*([+\-]?\d+[\d\.]*)\s*USDC", text)
                if m_p:
                    wallet_pnl = float(m_p.group(1))
            elif text.startswith("PnL:"):
                # Old format: "PnL: +52.3496 USDC (net +52.3403 after gas)"
                m_net = re.search(r"net\s*([+\-]?\d+[\d\.]*)\s*after gas", text)
                if m_net:
                    wallet_pnl = float(m_net.group(1))
                else:
                    m_p = re.search(r"PnL:\s*([+\-]?\d+[\d\.]*)\s*USDC", text)
                    if m_p:
                        wallet_pnl = float(m_p.group(1))

            # LP net PnL
            if "LP net PnL" in text:
                m_lp = re.search(r"LP net PnL.*:\s*([+\-]?\d+[\d\.]*)\s*USDC", text)
                if m_lp:
                    lp_pnl = float(m_lp.group(1))

            # Fees this cycle (sent)
            if "Fees this cycle" in text:
                m_f = re.search(r"Fees this cycle.*:\s*([+\-]?\d+[\d\.]*)\s*USDC", text)
                if m_f:
                    fees_sent = float(m_f.group(1))

            # Value in / out (several formats)
            if "Value in (LP):" in text or "Value in:" in text or "LP in:" in text:
                m_vi = re.search(r"\$ ?([\d,]+\.\d+)", text)
                if m_vi:
                    value_in = float(m_vi.group(1).replace(",", ""))

            if "Value out (LP):" in text or "Value out:" in text or "LP out:" in text:
                m_vo = re.search(r"\$ ?([\d,]+\.\d+)", text)
                if m_vo:
                    value_out = float(m_vo.group(1).replace(",", ""))

            # Exit reason
            if "Exit reason:" in text:
                exit_reason = text.split("Exit reason:")[-1].strip()
            else:
                m_r = re.search(r"reason:\s*([a-zA-Z0-9_]+)", text)
                if m_r:
                    exit_reason = m_r.group(1)

            j += 1

        cycle["duration_minutes"] = duration_minutes

        # Compute timestamp_create from duration
        if ts_exit and duration_minutes is not None:
            ts_create = ts_exit - timedelta(minutes=duration_minutes)
            cycle["timestamp_create"] = ts_create.isoformat() + "Z"
        else:
            cycle["timestamp_create"] = None

        # Map to fields expected by lp_analysis.py
        cycle["wallet_pnl_after_fees"] = wallet_pnl
        cycle["real_pnl_after_fees"] = wallet_pnl  # use wallet PnL as real realized PnL
        cycle["net_lp_pnl_after_gas"] = lp_pnl
        cycle["value_in_usd"] = value_in
        cycle["value_out_usd"] = value_out
        cycle["fees_sent_usdc"] = fees_sent
        cycle["exit_reason"] = exit_reason

        cycles.append(cycle)
        i = j

    return cycles


def main():
    lines = load_lines(INPUT_FILE)
    cycles = parse_cycles(lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in cycles:
            f.write(json.dumps(c) + "\n")

    print(f"[OK] Wrote {len(cycles)} cycles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
