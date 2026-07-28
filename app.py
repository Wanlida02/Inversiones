import os
import json
import sqlite3
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Valoración Internacional", layout="wide")
st.title("Valoración de Acciones - Arquitectura internacional")
st.caption("Finnhub como fuente principal de perfil, peers y series internacionales; Yahoo Finance como apoyo para precio histórico y validación.")

DB_PATH = "valuation_cache_intl.db"
FINNHUB_BASE = "https://finnhub.io/api/v1"
TEST_TICKERS = ["AAPL", "SAP", "NESN.SW"]


def get_secret_or_input(key_name, label, password=True):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return st.sidebar.text_input(label, type='password' if password else 'default')


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS company_profile (symbol TEXT PRIMARY KEY, name TEXT, country TEXT, currency TEXT, exchange TEXT, finnhub_industry TEXT, ipo TEXT, raw_json TEXT, updated_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS company_peers (symbol TEXT, peer TEXT, updated_at TEXT, PRIMARY KEY(symbol, peer))")
    conn.execute("CREATE TABLE IF NOT EXISTS sector_growth_proxy (anchor_symbol TEXT, peer_symbol TEXT, metric TEXT, year TEXT, value REAL, updated_at TEXT, PRIMARY KEY(anchor_symbol, peer_symbol, metric, year))")
    return conn


def finnhub_get(endpoint, params, token):
    params = dict(params)
    params['token'] = token
    r = requests.get(f"{FINNHUB_BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def cache_profile(symbol, payload):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO company_profile(symbol, name, country, currency, exchange, finnhub_industry, ipo, raw_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (symbol, payload.get('name'), payload.get('country'), payload.get('currency'), payload.get('exchange'), payload.get('finnhubIndustry'), payload.get('ipo'), json.dumps(payload), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_profile(symbol):
    conn = get_conn()
    row = conn.execute('SELECT name, country, currency, exchange, finnhub_industry, ipo, raw_json FROM company_profile WHERE symbol=?', (symbol,)).fetchone()
    conn.close()
    if not row:
        return None
    return {'name': row[0], 'country': row[1], 'currency': row[2], 'exchange': row[3], 'finnhubIndustry': row[4], 'ipo': row[5], 'raw_json': row[6], 'cached': True}


def get_profile(symbol, token):
    cached = load_profile(symbol)
    if cached:
        return cached
    payload = finnhub_get('stock/profile2', {'symbol': symbol}, token)
    cache_profile(symbol, payload)
    payload['cached'] = False
    return payload


def cache_peers(symbol, peers):
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    for p in peers:
        conn.execute('INSERT OR REPLACE INTO company_peers(symbol, peer, updated_at) VALUES (?, ?, ?)', (symbol, p, now))
    conn.commit()
    conn.close()


def load_peers(symbol):
    conn = get_conn()
    rows = conn.execute('SELECT peer FROM company_peers WHERE symbol=?', (symbol,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_peers(symbol, token):
    cached = load_peers(symbol)
    if cached:
        return cached
    payload = finnhub_get('stock/peers', {'symbol': symbol}, token)
    peers = [p for p in payload if isinstance(p, str) and p != symbol][:8]
    cache_peers(symbol, peers)
    return peers


def pick_row(df, names):
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return pd.Series(dtype=float)


def mean_growth_last_years(series, years=4):
    s = series.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) == 0:
        return np.nan
    return float(g.tail(min(years, len(g))).mean())


def yahoo_pack(symbol):
    tk = yf.Ticker(symbol)
    return {'info': tk.info, 'financials': tk.financials.copy(), 'cashflow': tk.cashflow.copy(), 'balance_sheet': tk.balance_sheet.copy(), 'history': tk.history(period='max', interval='1d', auto_adjust=False)}


def extract_metrics(ydata):
    info, inc, cf, bs = ydata['info'], ydata['financials'], ydata['cashflow'], ydata['balance_sheet']
    revenue = pick_row(inc, ['Total Revenue', 'Operating Revenue'])
    net_income = pick_row(inc, ['Net Income'])
    ocf = pick_row(cf, ['Operating Cash Flow'])
    capex = -np.abs(pick_row(cf, ['Capital Expenditure', 'Capital Expenditures']))
    common = ocf.index.intersection(capex.index)
    fcf = ocf[common] + capex[common] if len(common) else pd.Series(dtype=float)
    base_fcf = float(fcf.tail(2).mean()) if len(fcf) >= 2 else (float(fcf.iloc[-1]) if len(fcf) else np.nan)
    growth_real = mean_growth_last_years(net_income, 4)
    if not np.isfinite(growth_real):
        growth_real = mean_growth_last_years(revenue, 4)
    if not np.isfinite(growth_real):
        growth_real = 0.05
    growth_base = min(max(growth_real, 0.04), 0.20)
    debt = float(bs.loc['Total Debt'].dropna().iloc[0]) if 'Total Debt' in bs.index and len(bs.loc['Total Debt'].dropna()) else 0.0
    cash = 0.0
    for k in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Other Short Term Investments']:
        if k in bs.index and len(bs.loc[k].dropna()):
            cash += float(bs.loc[k].dropna().iloc[0])
    return {'price': info.get('currentPrice', info.get('regularMarketPrice', np.nan)), 'shares': info.get('sharesOutstanding', np.nan), 'revenue': revenue, 'net_income': net_income, 'base_fcf': base_fcf, 'growth_real': growth_real, 'growth_base': growth_base, 'debt': debt, 'cash': cash, 'forward_pe': info.get('forwardPE', np.nan)}


def dcf_intrinsic_from_base(base_fcf, growth_rate, shares, cash, debt, wacc=0.075, term_growth=0.03):
    if not np.isfinite(base_fcf) or base_fcf <= 0 or not np.isfinite(shares) or shares <= 0:
        return np.nan
    fcf_proj = []
    curr = base_fcf
    for _ in range(5):
        curr *= (1 + growth_rate)
        fcf_proj.append(curr)
    slowdown = growth_rate
    for _ in range(5):
        slowdown = max(slowdown * 0.8, 0.03)
        curr *= (1 + slowdown)
        fcf_proj.append(curr)
    pv_fcf = sum(f / ((1 + wacc) ** (i + 1)) for i, f in enumerate(fcf_proj))
    vt = (fcf_proj[-1] * (1 + term_growth)) / (wacc - term_growth)
    pv_vt = vt / ((1 + wacc) ** 10)
    ev = pv_fcf + pv_vt
    equity = ev + cash - debt
    return equity / shares


def intrinsic_path(base_fcf, growth_rate, shares, cash, debt, years=5):
    vals = {}
    curr_fcf, curr_debt = base_fcf, debt
    for y in range(years + 1):
        vals[y] = dcf_intrinsic_from_base(curr_fcf, max(growth_rate * (0.95 ** y), 0.03), shares, cash, curr_debt)
        curr_fcf *= (1 + growth_rate)
        curr_debt *= 0.99
    return pd.Series(vals)


def historical_pe_series(financials, history):
    eps = pick_row(financials, ['Diluted EPS', 'Basic EPS', 'EPS Diluted', 'EPS Basic'])
    if eps.empty or history.empty:
        return pd.Series(dtype=float)
    yearly_px = history['Close'].resample('YE').last()
    px_by_year = pd.Series(yearly_px.values, index=yearly_px.index.year)
    eps_by_year = pd.Series(eps.values, index=eps.index.year)
    common = sorted(set(px_by_year.index).intersection(set(eps_by_year.index)))
    out = {}
    for y in common:
        e = eps_by_year.loc[y]
        p = px_by_year.loc[y]
        if pd.notna(e) and e != 0 and pd.notna(p):
            out[y] = p / e
    return pd.Series(out).sort_index().tail(5)


def cache_growth_proxy(anchor_symbol, peer_symbol, metric, year, value):
    conn = get_conn()
    conn.execute('INSERT OR REPLACE INTO sector_growth_proxy(anchor_symbol, peer_symbol, metric, year, value, updated_at) VALUES (?, ?, ?, ?, ?, ?)', (anchor_symbol, peer_symbol, metric, str(year), float(value), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_growth_proxy(anchor_symbol, metric):
    conn = get_conn()
    df = pd.read_sql_query('SELECT peer_symbol, year, value FROM sector_growth_proxy WHERE anchor_symbol=? AND metric=? ORDER BY year', conn, params=(anchor_symbol, metric))
    conn.close()
    return df


def peer_growth_series(anchor_symbol, peers):
    cached = load_growth_proxy(anchor_symbol, 'revenue_growth')
    if not cached.empty:
        return cached
    rows = []
    for peer in peers[:6]:
        try:
            yd = yahoo_pack(peer)
            rev = pick_row(yd['financials'], ['Total Revenue', 'Operating Revenue']).tail(5)
            if len(rev) >= 2:
                g = rev.pct_change().dropna()
                for idx, val in g.items():
                    year = idx.year if hasattr(idx, 'year') else str(idx)
                    cache_growth_proxy(anchor_symbol, peer, 'revenue_growth', year, val)
                    rows.append({'peer_symbol': peer, 'year': str(year), 'value': float(val)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def aggregate_peer_growth(anchor_symbol, peers):
    df = peer_growth_series(anchor_symbol, peers)
    if df.empty:
        return pd.Series(dtype=float)
    agg = df.groupby('year')['value'].median().sort_index()
    agg.index = agg.index.astype(str)
    return agg


def run_diagnostics(token, tickers=TEST_TICKERS):
    rows = []
    for symbol in tickers:
        result = {'ticker': symbol}
        try:
            profile = get_profile(symbol, token)
            peers = get_peers(symbol, token)
            ydata = yahoo_pack(symbol)
            metrics = extract_metrics(ydata)
            pe_hist = historical_pe_series(ydata['financials'], ydata['history'])
            peer_growth = aggregate_peer_growth(symbol, peers)
            base_path = intrinsic_path(metrics['base_fcf'], metrics['growth_base'], metrics['shares'], metrics['cash'], metrics['debt'])
            result.update({
                'profile_ok': bool(profile.get('name') or profile.get('finnhubIndustry')),
                'peers_ok': len(peers) > 0,
                'financials_ok': len(metrics['revenue']) > 0,
                'pe_hist_points': len(pe_hist),
                'peer_growth_points': len(peer_growth),
                'valuation_ok': bool(np.isfinite(dcf_intrinsic_from_base(metrics['base_fcf'], metrics['growth_base'], metrics['shares'], metrics['cash'], metrics['debt']))),
                'path_points': len(base_path),
                'status': 'OK'
            })
        except Exception as e:
            result.update({'status': f'ERROR: {e}'})
        rows.append(result)
    return pd.DataFrame(rows)

st.sidebar.header('Credenciales y parámetros')
finnhub_token = get_secret_or_input('FINNHUB_API_KEY', 'Finnhub API Key')
ticker = st.sidebar.text_input('Ticker', 'AAPL').upper().strip()
custom_growth = st.sidebar.number_input('Crecimiento personalizado (%)', 0.0, 100.0, 10.0, 0.5) / 100.0
pess_cut = st.sidebar.slider('Recorte escenario pesimista (%)', 10, 90, 50) / 100.0
run = st.sidebar.button('Analizar')
run_tests = st.sidebar.button('Ejecutar diagnóstico')

if run_tests:
    if not finnhub_token:
        st.error('Necesitas API key de Finnhub para ejecutar el diagnóstico.')
    else:
        diag = run_diagnostics(finnhub_token)
        st.subheader('Diagnóstico automático')
        st.dataframe(diag, use_container_width=True)
        ok_count = int((diag['status'] == 'OK').sum()) if 'status' in diag.columns else 0
        st.caption(f'Tickers probados: {len(diag)} | Correctos: {ok_count}')

if run:
    if not finnhub_token:
        st.error('Necesitas API key de Finnhub. Puedes ponerla en la barra lateral o en Streamlit secrets como FINNHUB_API_KEY.')
    else:
        try:
            profile = get_profile(ticker, finnhub_token)
            peers = get_peers(ticker, finnhub_token)
            ydata = yahoo_pack(ticker)
            metrics = extract_metrics(ydata)
            currency = profile.get('currency') or ydata['info'].get('currency') or 'USD'
            name = profile.get('name') or ydata['info'].get('longName') or ticker
            industry = profile.get('finnhubIndustry') or ydata['info'].get('industry') or 'No disponible'
            country = profile.get('country') or 'No disponible'
            exchange = profile.get('exchange') or ydata['info'].get('exchange') or 'No disponible'
            g_base = metrics['growth_base']
            g_pes = g_base * (1 - pess_cut)
            g_cus = custom_growth
            v_base = dcf_intrinsic_from_base(metrics['base_fcf'], g_base, metrics['shares'], metrics['cash'], metrics['debt'])
            v_pes = dcf_intrinsic_from_base(metrics['base_fcf'], g_pes, metrics['shares'], metrics['cash'], metrics['debt'])
            v_cus = dcf_intrinsic_from_base(metrics['base_fcf'], g_cus, metrics['shares'], metrics['cash'], metrics['debt'])
            pe_hist = historical_pe_series(ydata['financials'], ydata['history'])
            peer_growth = aggregate_peer_growth(ticker, peers)
            company_rev_growth = metrics['revenue'].tail(5).pct_change().dropna() if len(metrics['revenue']) else pd.Series(dtype=float)
            company_rev_growth.index = [str(x.year) for x in company_rev_growth.index]
            base_path = intrinsic_path(metrics['base_fcf'], g_base, metrics['shares'], metrics['cash'], metrics['debt'])
            pes_path = intrinsic_path(metrics['base_fcf'], g_pes, metrics['shares'], metrics['cash'], metrics['debt'])
            cus_path = intrinsic_path(metrics['base_fcf'], g_cus, metrics['shares'], metrics['cash'], metrics['debt'])
            st.subheader(name)
            st.write(f"Industria Finnhub: {industry} | País: {country} | Bolsa: {exchange}")
            st.caption(f"Peers detectados por Finnhub: {', '.join(peers[:6]) if peers else 'No disponibles'}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Precio actual', f"{metrics['price']:.2f} {currency}")
            c2.metric('Valor base', f"{v_base:.2f} {currency}" if np.isfinite(v_base) else 'No disp.')
            c3.metric('Valor pesimista', f"{v_pes:.2f} {currency}" if np.isfinite(v_pes) else 'No disp.')
            c4.metric('Valor personalizado', f"{v_cus:.2f} {currency}" if np.isfinite(v_cus) else 'No disp.')
            st.subheader('PER histórico real')
            fig1 = go.Figure()
            if len(pe_hist):
                fig1.add_trace(go.Scatter(x=pe_hist.index.astype(str), y=pe_hist.values, mode='lines+markers', name='PER empresa'))
            fig1.update_layout(height=330, xaxis_title='Año', yaxis_title='PER')
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"Forward PER actual: {metrics['forward_pe'] if np.isfinite(metrics['forward_pe']) else 'No disponible'}")
            st.subheader('Crecimiento internacional: empresa vs peers')
            fig2 = go.Figure()
            if len(company_rev_growth):
                fig2.add_trace(go.Bar(x=company_rev_growth.index, y=company_rev_growth.values * 100, name='Empresa'))
            if len(peer_growth):
                fig2.add_trace(go.Scatter(x=list(peer_growth.index), y=peer_growth.values * 100, mode='lines+markers', name='Peers medianos'))
            fig2.update_layout(height=340, xaxis_title='Año', yaxis_title='Crecimiento ventas (%)')
            st.plotly_chart(fig2, use_container_width=True)
            st.caption('La estructura queda lista para evolucionar de peers a series sectoriales puras si incorporas una fuente adicional agregada internacional.')
            st.subheader('Precio vs valor relativo a 5 años')
            fig3 = go.Figure()
            years = list(base_path.index)
            fig3.add_trace(go.Scatter(x=years, y=[metrics['price']] * len(years), mode='lines+markers', name='Precio actual'))
            fig3.add_trace(go.Scatter(x=years, y=base_path.values, mode='lines+markers', name='Valor base'))
            fig3.add_trace(go.Scatter(x=years, y=pes_path.values, mode='lines+markers', name='Valor pesimista'))
            fig3.add_trace(go.Scatter(x=years, y=cus_path.values, mode='lines+markers', name='Valor personalizado'))
            fig3.update_layout(height=380, xaxis_title='Años desde hoy', yaxis_title=f'Valor ({currency})')
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader('Arquitectura de datos')
            st.markdown('- Fuente principal internacional: **Finnhub** para perfil e identificación de peers.')
            st.markdown('- Crecimiento internacional agregado: **mediana de peers detectados por Finnhub**, cacheada en SQLite.')
            st.markdown('- Precios, EPS y estados financieros: **Yahoo Finance / yfinance**.')
            st.markdown(f'- Base local: **SQLite ({DB_PATH})**.')
            st.markdown('- Credencial: API key desde **Streamlit secrets** o input lateral dentro de la propia app.')
        except Exception as e:
            st.error(f'Error al analizar {ticker}: {e}')
