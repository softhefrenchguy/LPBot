import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Area,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './index.css'

const API_BASE = import.meta.env.VITE_API_BASE || `${window.location.protocol}//${window.location.hostname}:8000`
const WS_BASE = API_BASE.replace(/^http/, 'ws')

const pct = (v, digits = 2) => `${((Number(v) || 0) * 100).toFixed(digits)}%`
const usd = (v, digits = 0) => `$${(Number(v) || 0).toLocaleString(undefined, { maximumFractionDigits: digits })}`
const num = (v, digits = 2) => (Number(v) || 0).toLocaleString(undefined, { maximumFractionDigits: digits })
const regimeClass = (r) => r === 'BULL' ? 'text-mint bg-mint/10' : r === 'BEAR' ? 'text-danger bg-danger/10' : 'text-amber bg-amber/10'
const returnClass = (v) => Number(v) >= 0 ? 'text-mint' : 'text-danger'

function useDashboardData() {
  const [status, setStatus] = useState(null)
  const [performance, setPerformance] = useState([])
  const [trades, setTrades] = useState([])
  const [markets, setMarkets] = useState({})
  const [news, setNews] = useState([])
  const [error, setError] = useState('')

  const load = async () => {
    try {
      const [s, p, t, m, n] = await Promise.all([
        fetch(`${API_BASE}/api/status`).then(r => r.json()),
        fetch(`${API_BASE}/api/performance`).then(r => r.json()),
        fetch(`${API_BASE}/api/trades`).then(r => r.json()),
        fetch(`${API_BASE}/api/markets`).then(r => r.json()),
        fetch(`${API_BASE}/api/news`).then(r => r.json()),
      ])
      setStatus(s); setPerformance(p); setTrades(t); setMarkets(m); setNews(n); setError('')
    } catch (e) {
      setError(e.message || 'API unavailable')
    }
  }

  useEffect(() => {
    load()
    const timer = setInterval(load, 60000)
    let ws
    try {
      ws = new WebSocket(`${WS_BASE}/ws/live`)
      ws.onmessage = (event) => setStatus(prev => ({ ...(prev || {}), ...JSON.parse(event.data) }))
    } catch (_) {}
    return () => { clearInterval(timer); if (ws) ws.close() }
  }, [])

  return { status, performance, trades, markets, news, error }
}

function Header({ status, error }) {
  const healthy = status?.healthy && !error
  return (
    <header className="mb-8 flex flex-col gap-4 rounded-[2rem] border border-mint/10 bg-ink/50 p-5 shadow-glow md:flex-row md:items-center md:justify-between">
      <div>
        <div className="font-display text-3xl font-bold tracking-tight">LPBot</div>
        <div className="mt-1 text-sm text-emerald-100/55">Day {status?.days_live ?? '--'} | Last updated: {status?.last_updated ? new Date(status.last_updated).toLocaleString() : 'loading'}</div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <span className={`badge rounded-full px-4 py-2 text-sm ${healthy ? 'text-mint' : 'text-danger'}`}>
          <span className={`mr-2 inline-block h-2 w-2 rounded-full ${healthy ? 'bg-mint' : 'bg-danger'}`} />
          {healthy ? 'LIVE / PAPER' : 'REVIEW'}
        </span>
        <span className="badge rounded-full px-4 py-2 text-sm text-emerald-100/70">{status?.flags || 'No flags'}</span>
      </div>
    </header>
  )
}

function EquityChart({ data }) {
  const chartData = useMemo(() => data.map(d => ({
    date: d.date,
    'LPBot Strategy': (Number(d.strategy_ret) || 0) * 100,
    'ETH Spot': (Number(d.eth_spot) || 0) * 100,
    '50/50 Basket': (Number(d.combined) || 0) * 100,
  })), [data])

  return (
    <section className="glass rounded-[2rem] p-5">
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <h2 className="font-display text-xl font-semibold">Equity Curve</h2>
        <div className="flex flex-wrap gap-3 text-sm text-emerald-100/65">
          <span><span className="mr-2 inline-block h-2 w-5 rounded-full bg-[#60a5fa]" />LPBot Strategy</span>
          <span><span className="mr-2 inline-block h-2 w-5 rounded-full bg-[#fb923c]" />ETH Spot</span>
          <span><span className="mr-2 inline-block h-2 w-5 rounded-full border border-[#94a3b8]" />50/50 Basket</span>
        </div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ left: 0, right: 18, top: 10, bottom: 0 }}>
            <CartesianGrid stroke="#20352f" strokeDasharray="3 3" />
            <XAxis dataKey="date" stroke="#7f968d" tick={{ fontSize: 11 }} />
            <YAxis stroke="#7f968d" tickFormatter={(v) => `${v.toFixed(0)}%`} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#0f1b18', border: '1px solid #20352f', borderRadius: 16 }} formatter={(v) => `${Number(v).toFixed(2)}%`} />
            <Line type="monotone" dataKey="LPBot Strategy" stroke="#60a5fa" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="ETH Spot" stroke="#fb923c" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="50/50 Basket" stroke="#94a3b8" strokeWidth={2} strokeDasharray="6 6" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}

function BenchmarksPanel({ status, performance, trades }) {
  const strategy = Number(status?.strategy_return) || 0
  const eth = Number(status?.eth_spot) || 0
  const btc = Number(status?.btc_spot) || 0
  const basket = Number(status?.basket_spot) || (strategy - (Number(status?.excess) || 0))
  const rows = [
    ['Strategy', strategy, null],
    ['ETH spot', eth, strategy - eth],
    ['BTC spot', btc, strategy - btc],
    ['50/50 basket', basket, strategy - basket],
  ]
  const daily = useMemo(() => {
    const vals = (performance || []).map((d) => Number(d.strategy_ret) || 0)
    return vals.slice(1).map((v, i) => ((1 + v) / Math.max(1e-9, 1 + vals[i])) - 1).filter(Number.isFinite)
  }, [performance])
  const mean = daily.reduce((a, b) => a + b, 0) / Math.max(1, daily.length)
  const variance = daily.reduce((a, b) => a + ((b - mean) ** 2), 0) / Math.max(1, daily.length - 1)
  const annVol = Math.sqrt(Math.max(0, variance)) * Math.sqrt(365)
  const wins = trades.filter(t => Number(t.return_pct) > 0).length
  const losses = trades.filter(t => Number(t.return_pct) <= 0).length
  const winRate = trades.length ? wins / trades.length : 0

  return (
    <section className="glass rounded-[2rem] p-5">
      <div className="mb-4">
        <h2 className="font-display text-xl font-semibold">Strategy vs Benchmarks</h2>
        <p className="text-sm text-emerald-100/45">Since Mar 21</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-emerald-100/45"><tr><th className="py-2">Benchmark</th><th>Return</th><th>vs Strategy</th></tr></thead>
            <tbody>{rows.map(([label, ret, diff]) => (
              <tr key={label} className="border-t border-line/80">
                <td className="py-3 font-semibold">{label}</td>
                <td className={returnClass(ret)}>{pct(ret)}</td>
                <td className={diff === null ? 'text-emerald-100/35' : 'text-mint'}>{diff === null ? '-' : pct(diff)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div className="rounded-2xl border border-line bg-white/[0.02] p-4">
          <h3 className="mb-3 font-display text-lg font-semibold">Risk Metrics</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Info label="Peak DD" value={pct(status?.peak_dd)} className="text-amber" />
            <Info label="Ann Vol" value={pct(annVol)} />
            <Info label="Days live" value={status?.days_live ?? '--'} />
            <Info label="Trades" value={trades.length} />
            <Info label="Win rate" value={`${pct(winRate, 0)} (${wins} wins, ${losses} losses)`} span />
          </div>
        </div>
      </div>
    </section>
  )
}

function AssetCard({ name, status, btc = false }) {
  return (
    <div className="glass rounded-[2rem] p-5">
      <div className="flex items-start justify-between">
        <h3 className="font-display text-2xl font-semibold">{name}</h3>
        <span className={`rounded-full px-3 py-1 text-sm ${regimeClass(btc ? status?.btc_regime : status?.eth_regime)}`}>{btc ? status?.btc_regime : status?.eth_regime}</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <Info label="Price" value={usd(btc ? status?.btc_price : status?.eth_price)} />
        <Info label="24h" value={pct(btc ? status?.btc_24h_pct : status?.eth_24h_pct, 1)} className={returnClass(btc ? status?.btc_24h_pct : status?.eth_24h_pct)} />
        <Info label={btc ? 'EMA15/40/120' : 'EMA21/55/144'} value={btc ? `${num(status?.btc_ema15,0)} / ${num(status?.btc_ema40,0)} / ${num(status?.btc_ema120,0)}` : `${num(status?.ema21,0)} / ${num(status?.ema55,0)} / ${num(status?.ema144,0)}`} span />
        <Info label="Stack aligned" value={(btc ? status?.btc_stack_aligned : status?.stack_aligned) ? 'YES' : 'NO'} />
        <Info label="Weight" value={pct(btc ? status?.btc_weight : status?.eth_weight, 0)} />
        {!btc && <Info label="Vol regime" value={`${status?.vol_regime || 'NA'} (${pct(status?.vol_percentile,0)})`} />}
        <Info label="Conviction" value={btc ? status?.btc_conviction : status?.conviction} />
      </div>
    </div>
  )
}

function Info({ label, value, className = '', span = false }) {
  return <div className={span ? 'col-span-2' : ''}><div className="text-emerald-100/45">{label}</div><div className={`font-semibold ${className}`}>{value}</div></div>
}

function MarketsTable({ markets }) {
  const rows = Object.entries(markets || {})
  return (
    <section className="glass rounded-[2rem] p-5">
      <h2 className="mb-4 font-display text-xl font-semibold">Markets</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-emerald-100/45"><tr><th className="py-2">Asset</th><th>Price</th><th>24h%</th><th>Regime</th><th>Trend</th></tr></thead>
          <tbody>{rows.map(([asset, m]) => (
            <tr key={asset} className="border-t border-line/80">
              <td className="py-3 font-semibold">{asset}</td>
              <td>{usd(m.price, asset.includes('DXY') ? 1 : 2)}</td>
              <td className={returnClass(m.pct)}>{pct(m.pct, 1)}</td>
              <td><span className={`rounded-full px-2 py-1 ${regimeClass(m.regime)}`}>{m.regime}</span></td>
              <td>{m.stack_aligned ? <span className="text-mint">↑ stack aligned</span> : <span className="text-emerald-100/35">--</span>}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  )
}

function Trades({ trades }) {
  return (
    <section className="glass rounded-[2rem] p-5">
      <h2 className="mb-4 font-display text-xl font-semibold">Trade History</h2>
      {trades.length === 0 ? <div className="text-emerald-100/45">No completed trades logged.</div> : <div className="overflow-x-auto"><table className="w-full text-left text-sm">
        <thead className="text-emerald-100/45"><tr><th className="py-2">Entry</th><th>Exit</th><th>Return</th><th>Days</th><th>Reason</th></tr></thead>
        <tbody>{trades.map((t, i) => <tr key={i} className="border-t border-line/80"><td className="py-3">{t.entry_date}</td><td>{t.exit_date}</td><td className={returnClass(t.return_pct)}>{pct(t.return_pct)}</td><td>{t.days_held}</td><td>{t.exit_reason}</td></tr>)}</tbody>
      </table></div>}
    </section>
  )
}

function NewsFeed({ news }) {
  return (
    <section className="glass rounded-[2rem] p-5">
      <h2 className="mb-4 font-display text-xl font-semibold">News Feed</h2>
      <div className="space-y-3">{news.map((n, i) => <details key={i} open={i === news.length - 1} className={`rounded-2xl border p-4 ${n.major_event ? 'border-amber/30 bg-amber/10' : 'border-line bg-white/[0.02]'}`}>
        <summary className="cursor-pointer font-semibold">{n.date} | {n.major_event ? 'Major event' : 'No major macro event'} | {n.direction}</summary>
        <p className="mt-2 text-sm text-emerald-100/70">{n.summary}</p>
      </details>)}</div>
    </section>
  )
}

function GoldCard({ status }) {
  return (
    <section className="glass rounded-[2rem] p-5">
      <h2 className="mb-4 font-display text-xl font-semibold">Gold Sleeve Status</h2>
      <div className="grid gap-3 text-sm md:grid-cols-4">
        <Info label="PAXG" value={`${usd(status?.gold_price, 2)} (${pct(status?.gold_24h_pct, 1)})`} />
        <Info label="EMA21/55/144" value={`${num(status?.gold_ema21,0)} / ${num(status?.gold_ema55,0)} / ${num(status?.gold_ema144,0)}`} span />
        <Info label="Gate" value={status?.gold_flat_bear_gate ? 'OPEN' : 'CLOSED'} className={status?.gold_flat_bear_gate ? 'text-mint' : 'text-emerald-100/45'} />
        <Info label="Condition" value={status?.gold_condition_met ? 'YES' : 'NO'} className={status?.gold_condition_met ? 'text-mint' : 'text-danger'} />
      </div>
    </section>
  )
}

function OptionsVolCard({ status }) {
  const atmIv = Number(status?.dvol_atm_iv_30d)
  const ivPct = Number(status?.dvol_iv_percentile)
  const termSlope = Number(status?.dvol_term_slope)
  const historyDays = Number(status?.dvol_history_days) || 0
  const hasIv = Number.isFinite(atmIv)
  const hasPct = Number.isFinite(ivPct)
  const hasSlope = Number.isFinite(termSlope)
  const insufficient = Boolean(status?.dvol_insufficient_history)

  return (
    <section className="glass rounded-[2rem] p-5">
      <h2 className="mb-4 font-display text-xl font-semibold">Options Vol Signal</h2>
      <div className="grid gap-3 text-sm md:grid-cols-4">
        <Info label="ATM IV 30d" value={hasIv ? `${atmIv.toFixed(1)}%` : 'NA'} />
        <Info label="IV percentile" value={hasPct ? pct(ivPct, 0) : 'Building'} />
        <Info label="Options regime" value={status?.dvol_options_vol_regime || 'NA'} />
        <Info label="RV regime" value={status?.dvol_rv_vol_regime || 'NA'} />
        <Info label="Term slope" value={hasSlope ? `${termSlope.toFixed(1)} vol pts` : 'NA'} className={hasSlope && termSlope < 0 ? 'text-amber' : ''} />
        <Info label="Agreement" value={status?.dvol_agreement || 'NA'} />
        <Info label="History" value={`${historyDays}/30 days`} />
        <Info label="Mode" value={insufficient ? 'LOG ONLY' : 'READY'} className={insufficient ? 'text-amber' : 'text-mint'} />
      </div>
    </section>
  )
}

function App() {
  const { status, performance, trades, markets, news, error } = useDashboardData()
  return (
    <main className="mx-auto max-w-7xl px-4 py-6 md:px-8">
      <Header status={status} error={error} />
      {error && <div className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 p-4 text-danger">API error: {error}</div>}
      <div className="mb-6"><EquityChart data={performance} /></div>
      <div className="mb-6"><BenchmarksPanel status={status || {}} performance={performance} trades={trades} /></div>
      <section className="mb-6 grid gap-4 lg:grid-cols-2"><AssetCard name="ETH" status={status || {}} /><AssetCard name="BTC" status={status || {}} btc /></section>
      <div className="mb-6"><MarketsTable markets={markets} /></div>
      <section className="mb-6 grid gap-4 lg:grid-cols-2"><Trades trades={trades} /><NewsFeed news={news} /></section>
      <section className="grid gap-4 lg:grid-cols-2"><GoldCard status={status || {}} /><OptionsVolCard status={status || {}} /></section>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
