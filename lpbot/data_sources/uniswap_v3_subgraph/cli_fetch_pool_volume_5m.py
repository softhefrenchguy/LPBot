from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


def _parse_date_utc(val: str) -> pd.Timestamp:
    ts = pd.to_datetime(val, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def gql_post(url: str, query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if "errors" in data:
        raise SystemExit(f"GraphQL errors: {data['errors']}")
    return data


def introspect_query_fields(url: str) -> set[str]:
    query = "{ __schema { queryType { fields { name } } } }"
    data = gql_post(url, query)
    fields = data.get("data", {}).get("__schema", {}).get("queryType", {}).get("fields", [])
    return {f.get("name") for f in fields if isinstance(f, dict) and f.get("name")}


def _candidate_type_names(entity: str) -> list[str]:
    def singularize(name: str) -> str:
        if name.endswith("ies"):
            return name[:-3] + "y"
        if name.endswith("Datas"):
            return name[:-1]
        if name.endswith("Snapshots"):
            return name[:-1]
        if name.endswith("s"):
            return name[:-1]
        return name

    names: list[str] = []
    for base in {entity, entity[:1].upper() + entity[1:]}:
        names.append(base)
        names.append(singularize(base))
    seen = set()
    ordered: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _unwrap_type_name(type_obj: dict | None) -> str | None:
    if not type_obj:
        return None
    name = type_obj.get("name")
    kind = type_obj.get("kind")
    of_type = type_obj.get("ofType")
    if name:
        return name
    if kind == "NON_NULL" and of_type:
        return _unwrap_type_name(of_type)
    return None


def _introspect_entity_fields(url: str, entity_name: str) -> tuple[str, set[str], dict[str, str]]:
    for type_name in _candidate_type_names(entity_name):
        query = (
            f'{{ __type(name: "{type_name}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} }} }}'
        )
        data = gql_post(url, query)
        type_obj = data.get("data", {}).get("__type")
        if type_obj is None:
            continue
        fields = type_obj.get("fields", [])
        field_names = set()
        field_types: dict[str, str] = {}
        for f in fields:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            name = f.get("name")
            field_names.add(name)
            type_name_inner = _unwrap_type_name(f.get("type"))
            if type_name_inner:
                field_types[name] = type_name_inner
        if field_names:
            return type_name, field_names, field_types
    raise SystemExit(f"Could not resolve GraphQL type for entity {entity_name}.")


def _fetch_page(
    url: str,
    entity: str,
    pool: str,
    time_field: str,
    volume_field: str,
    tvl_field: str | None,
    cursor: int,
    cursor_type: str,
    retries: int,
    retry_sleep: float,
) -> list[dict]:
    extra_fields = f" {tvl_field}" if tvl_field else ""
    query = (
        f'query($pool: String!, $cursor: {cursor_type}!) {{ {entity}('
        f'first: 1000, orderBy: {time_field}, orderDirection: desc, '
        f'where: {{ pool: $pool, {time_field}_lt: $cursor }}'
        f") {{ {time_field} {volume_field}{extra_fields} }} }}"
    )

    attempt = 0
    while True:
        cursor_val = str(cursor) if cursor_type == "BigInt" else int(cursor)
        data = gql_post(url, query, {"pool": pool, "cursor": cursor_val})
        rows = data.get("data", {}).get(entity, [])
        if rows is None:
            rows = []
        if rows:
            return rows
        if attempt < retries:
            attempt += 1
            time.sleep(retry_sleep)
            continue
        return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subgraph-url",
        required=False,
        help=(
            "GraphQL endpoint URL (e.g. Graph gateway URL for subgraph ID "
            "FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX)."
        ),
    )
    p.add_argument("--graph-api-key", default=None, help="The Graph gateway API key.")
    p.add_argument(
        "--gateway-base",
        default="https://gateway.thegraph.com/api",
        help="Gateway base URL (default: https://gateway.thegraph.com/api).",
    )
    p.add_argument(
        "--subgraph-id",
        default="FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX",
        help="Subgraph ID (default: Arbitrum Uniswap v3).",
    )
    p.add_argument(
        "--pool",
        required=True,
        help="Pool address (e.g. 0xC6962004f452bE9203591991D15f6b388e09E8D0).",
    )
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-out", default=None)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=1.5)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.bar_minutes != 5:
        raise SystemExit("Only --bar-minutes 5 is supported for now.")

    if args.subgraph_url:
        subgraph_url = args.subgraph_url
    elif args.graph_api_key:
        if not args.subgraph_id:
            raise SystemExit("Missing --subgraph-id for gateway URL construction.")
        subgraph_url = (
            f"{args.gateway_base}/{args.graph_api_key}/subgraphs/id/{args.subgraph_id}"
        )
    else:
        raise SystemExit("Either --subgraph-url or --graph-api-key must be provided.")

    display_url = subgraph_url
    if args.graph_api_key and args.graph_api_key in display_url:
        masked = args.graph_api_key[:4] + "****"
        display_url = display_url.replace(args.graph_api_key, masked)
    print(f"subgraph_url={display_url}")

    start_ts = _parse_date_utc(args.start)
    end_ts = _parse_date_utc(args.end)
    if end_ts <= start_ts:
        raise SystemExit("--end must be after --start.")

    query_fields = introspect_query_fields(subgraph_url)
    candidates = [
        "poolHourDatas",
        "poolHourData",
        "poolHourlyDatas",
        "poolHourlyData",
        "poolHourlySnapshots",
        "liquidityPoolHourlySnapshots",
        "poolHourSnapshots",
    ]
    entity = next((c for c in candidates if c in query_fields), None)
    if entity is None:
        keywords = ("pool", "hour", "day", "swap", "volume", "snapshot")
        filtered = sorted([f for f in query_fields if any(k in f.lower() for k in keywords)])
        raise SystemExit(f"No supported hourly entity found. Available: {filtered}")

    type_name, entity_fields, entity_field_types = _introspect_entity_fields(
        subgraph_url, entity
    )
    time_candidates = ["periodStartUnix", "timestamp", "hourStartUnix", "date"]
    time_field = next((c for c in time_candidates if c in entity_fields), None)
    if time_field is None:
        raise SystemExit(f"No supported time field found on {entity}. Fields: {sorted(entity_fields)}")

    volume_candidates = [
        "hourlyVolumeUSD",
        "volumeUSD",
        "volumeUsd",
        "cumulativeVolumeUSD",
        "hourlyVolumeByTokenUSD",
        "volumeUSDUntracked",
        "volume",
    ]
    volume_field = next((c for c in volume_candidates if c in entity_fields), None)
    if volume_field is None:
        raise SystemExit(f"No volume field found on {entity}. Fields: {sorted(entity_fields)}")
    if "usd" not in volume_field.lower():
        raise SystemExit(f"Volume field {volume_field} is not USD; only USD volume supported.")
    tvl_field = "tvlUSD" if "tvlUSD" in entity_fields else None

    if args.debug:
        print(f"query_fields={sorted(query_fields)}")
        print(f"type_name={type_name}")
        print(f"{entity}_fields={sorted(entity_fields)}")
        print(f"time_field_type={entity_field_types.get(time_field)}")

    print(f"entity_used={entity} time_field={time_field} volume_field={volume_field}")

    cursor_type = entity_field_types.get(time_field, "Int")
    if cursor_type not in {"Int", "BigInt"}:
        cursor_type = "BigInt" if "big" in cursor_type.lower() else "Int"

    pool = args.pool.lower()

    verify_field = None
    if "liquidityPool" in query_fields:
        verify_field = "liquidityPool"
    elif "pool" in query_fields:
        verify_field = "pool"

    if verify_field:
        verify_query = f"query($pool: String!) {{ {verify_field}(id: $pool) {{ id }} }}"
        verify_data = gql_post(subgraph_url, verify_query, {"pool": pool})
        found = verify_data.get("data", {}).get(verify_field)
        if not found or not found.get("id"):
            raise SystemExit(
                f"Pool not found in subgraph ({verify_field}). Check network/subgraph/pool."
            )
    elif args.debug:
        print("pool verification skipped (no pool entity in query fields)")
    cursor = int(end_ts.timestamp()) + 3600
    start_unix = int(start_ts.timestamp())

    hours: list[tuple[int, float, float]] = []
    pages = 0
    total = 0

    while True:
        rows = _fetch_page(
            subgraph_url,
            entity,
            pool,
            time_field,
            volume_field,
            tvl_field,
            cursor,
            cursor_type,
            args.retries,
            args.retry_sleep,
        )
        if not rows:
            break
        pages += 1
        total += len(rows)

        min_ts = None
        for row in rows:
            try:
                ts = int(row[time_field])
                vol = float(row.get(volume_field) or 0.0)
                tvl = float(row.get(tvl_field) or 0.0) if tvl_field else 0.0
            except Exception:
                continue
            if ts < start_unix:
                continue
            if ts >= int(end_ts.timestamp()):
                continue
            hours.append((ts, vol, tvl))
            if min_ts is None or ts < min_ts:
                min_ts = ts

        if min_ts is None:
            break
        cursor = int(min_ts)
        if cursor <= start_unix:
            break
        print(f"pages={pages} rows={total} oldest_ts={min_ts}")

    if not hours:
        raise SystemExit("No hourly rows fetched for the requested range.")

    raw_df = pd.DataFrame(hours, columns=["timestamp_hour", "volume_usd_hour", "tvlUSD"])
    raw_df["timestamp_hour"] = pd.to_datetime(raw_df["timestamp_hour"], unit="s", utc=True)
    raw_df = raw_df.sort_values("timestamp_hour").reset_index(drop=True)
    raw_df = raw_df[
        (raw_df["timestamp_hour"] >= start_ts) & (raw_df["timestamp_hour"] < end_ts)
    ]

    if args.raw_out is not None:
        raw_path = Path(args.raw_out)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df = raw_df.assign(entity_used=entity, time_field=time_field, volume_field=volume_field)
        raw_df.to_csv(raw_path, index=False)

    if args.bar_minutes != 5:
        raise SystemExit("Only --bar-minutes 5 is supported for now.")

    expanded_rows: list[tuple[pd.Timestamp, float]] = []
    for _, row in raw_df.iterrows():
        base_ts = row["timestamp_hour"]
        vol = float(row["volume_usd_hour"]) / 12.0
        for k in range(12):
            expanded_rows.append((base_ts + pd.Timedelta(minutes=5 * k), vol))

    vol_df = pd.DataFrame(expanded_rows, columns=["timestamp", "volume_usd"])

    full_index = pd.date_range(start=start_ts, end=end_ts, freq="5min", inclusive="left")
    vol_df = vol_df.set_index("timestamp").groupby(level=0)["volume_usd"].sum()
    vol_df = vol_df.reindex(full_index).fillna(0.0)
    vol_df = vol_df.reset_index().rename(columns={"index": "timestamp"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol_df.to_csv(out_path, index=False)

    print(f"rows_fetched={len(raw_df)}")
    print(f"range_start={start_ts} range_end={end_ts}")
    if not vol_df.empty:
        print(f"volume_median={float(vol_df['volume_usd'].median()):.6f}")
        print(f"volume_max={float(vol_df['volume_usd'].max()):.6f}")
    print(f"rows={len(vol_df)}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
