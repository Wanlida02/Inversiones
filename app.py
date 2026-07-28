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
st.title("Valoración de Acciones - Alpha Vantage + Finnhub")
st.caption("Finnhub para perfil y peers; Alpha Vantage para estados financieros históricos; Yahoo Finance solo para precios con caché.")

DB_PATH = "valuation_cache_intl.db"
FINNHUB_BASE = "https://finnhub.io/api/v1"
ALPHAV_BASE = "https://www.alphavantage.co/query"
TEST_TICKERS = ["AAPL", "SAP", "NESN.SW"]


def get_secret_or_input(key_name, label, password=True):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return st.sidebar.text_input(label, type='password' if password else 'default')


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS company_profile (symbol TEXT PRIMARY KEY, name TEXT, country TEXT, currency TEXT, exchange TEXT, finnhub_industry TEXT, ipo TEXT, raw_json TEXT, updated_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS company_peers (symbol TEXT, peer TEXT, updated_at TEXT, PRIMARY KEY(symbol, peer))")
    return conn


def finnhub_get(endpoint, params, token):
    params = dict(params)
    params['token'] = token
    r = requests.get(f"{FINNHUB_BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def alphav_get(params, api_key):
    p = dict(params)
    p['apikey'] = api_key
    r = requests.get(ALPHAV_BASE, params=p, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if 'Note' in data:
            raise ValueError(f"Alpha Vantage límite alcanzado: {data['Note']}")
        if 'Information' in data:
            raise ValueError(f"Alpha Vantage información: {data['Information']}")
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    return data


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
    peers = [p for p in payload if isinstance(p, str) and p != symbol][:4]
    cache_peers(symbol, peers)
    return peers


def ensure_datetime_index(series):
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.copy()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        return pd.Series(dtype=float)
    return s.sort_index()


def mean_growth_last_years(series, years=4):
    s = ensure_datetime_index(series).dropna().sort_index()
    if len(s) < 2:
        return np.nan
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) == 0:
        return np.nan
    return float(g.tail(min(years, len(g))).mean())


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_price_history(symbol):
    tk = yf.Ticker(symbol)
    hist = tk.history(period='max', interval='1d', auto_adjust=False)
    info = tk.info
    return info, hist


@st.cache_data(ttl=3600, show_spinner=False)
def alphav_financials(symbol, api_key):
    inc = alphav_get({'function': 'INCOME_STATEMENT', 'symbol': symbol}, api_key)
    bs = alphav_get({'function': 'BALANCE_SHEET', 'symbol': symbol}, api_key)
    cf = alphav_get({'function': 'CASH_FLOW', 'symbol': symbol}, api_key)
    earnings = alphav_get({'function': 'EARNINGS', 'symbol': symbol}, api_key)

    def to_series_annual(reports, key):
        vals = {}
        for r in reports:
            if 'fiscalDateEnding' in r and key in r and r[key] not in (None, '', 'None'):
                try:
                    y = int(r['fiscalDateEnding'][:4])
                    vals[y] = float(r[key])
                except Exception:
                    continue
        if not vals:
            return pd.Series(dtype=float)
        s = pd.Series(vals)
        s.index = pd.to_datetime([f"{y}-12-31" for y in s.index])
        return s.sort_index()

    inc_ann = inc.get('annualReports', []) if isinstance(inc, dict) else []
    bs_ann = bs.get('annualReports', []) if isinstance(bs, dict) else []
    cf_ann = cf.get('annualReports', []) if isinstance(cf, dict) else []
    earn_ann = earnings.get('annualEarnings', []) if isinstance(earnings, dict) else []

    revenue = to_series_annual(inc_ann, 'totalRevenue')
    net_income = to_series_annual(inc_ann, 'netIncome')
    ebitda = to_series_annual(inc_ann, 'ebitda')
    total_assets = to_series_annual(bs_ann, 'totalAssets')
    total_liab = to_series_annual(bs_ann, 'totalLiabilities')
    ocf = to_series_annual(cf_ann, 'operatingCashflow')
    capex = to_series_annual(cf_ann, 'capitalExpenditures')
    eps = to_series_annual(earn_ann, 'reportedEPS')

    capex = -capex.abs() if len(capex) else pd.Series(dtype=float)
    common = ocf.index.intersection(capex.index)
    fcf = ocf[common] + capex[common] if len(common) else pd.Series(dtype=float)

    base_fcf = float(fcf.tail(2).mean()) if len(fcf) >= 2 else (float(fcf.iloc[-1]) if len(fcf) else np.nan)
    growth_sales = mean_growth_last_years(revenue, 4)
    growth_real = mean_growth_last_years(net_income, 4)
    if not np.isfinite(growth_real):
        growth_real = growth_sales
    if not np.isfinite(growth_real):
        growth_real = 0.05
    growth_base = min(max(growth_real, 0.04), 0.20)

    invested_capital = None
    if not total_assets.empty and not total_liab.empty:
        common_bs = total_assets.index.intersection(total_liab.index)
        invested_capital = total_assets[common_bs] - total_liab[common_bs]

    return {
        'revenue': ensure_datetime_index(revenue),
        'net_income': ensure_datetime_index(net_income),
        'ebitda': ensure_datetime_index(ebitda),
        'eps': ensure_datetime_index(eps),
        'total_assets': ensure_datetime_index(total_assets),
        'total_liab': ensure_datetime_index(total_liab),
        'invested_capital': ensure_datetime_index(invested_capital) if invested_capital is not None else None,
        'ocf': ensure_datetime_index(ocf),
        'capex': ensure_datetime_index(capex),
        'fcf': ensure_datetime_index(fcf),
        'base_fcf': base_fcf,
        'growth_sales': growth_sales,
        'growth_real': growth_real,
        'growth_base': growth_base,
    }


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


def historical_pe_series(financials_eps, price_hist, max_years=10):
    eps = ensure_datetime_index(financials_eps)
    if eps.empty or price_hist.empty:
        return pd.Series(dtype=float)
    yearly_px = price_hist['Close'].resample('YE').last()
    px_by_year = pd.Series(yearly_px.values, index=yearly_px.index.year)
    eps_by_year = pd.Series(eps.values, index=eps.index.year)
    common = sorted(set(px_by_year.index).intersection(set(eps_by_year.index)))
    out = {}
    for y in common:
        e = eps_by_year.loc[y]
        p = px_by_year.loc[y]
        if pd.notna(e) and e != 0 and pd.notna(p):
            out[y] = p / e
    s = pd.Series(out).sort_index()
    return s.tail(max_years)


def metrics_table_from_2020(fin, shares):
    revenue = ensure_datetime_index(fin['revenue'])
    ebitda = ensure_datetime_index(fin['ebitda'])
    net_income = ensure_datetime_index(fin['net_income'])
    eps = ensure_datetime_index(fin['eps'])
    fcf = ensure_datetime_index(fin['fcf'])
    invested_capital = ensure_datetime_index(fin['invested_capital']) if fin['invested_capital'] is not None else None
    rows = []
    current_year = datetime.now().year
    for year in range(2020, current_year + 1):
        row = {'Year': year, 'ROIC_%': np.nan, 'EBITDA_margin_%': np.nan, 'EPS': np.nan, 'Shares': np.nan, 'FCF': np.nan}
        if year in revenue.index.year and year in ebitda.index.year:
            e = float(ebitda[ebitda.index.year == year].iloc[-1])
            r = float(revenue[revenue.index.year == year].iloc[-1])
            if r != 0:
                row['EBITDA_margin_%'] = e / r * 100
        if invested_capital is not None and year in invested_capital.index.year and year in net_income.index.year:
            ic = float(invested_capital[invested_capital.index.year == year].iloc[-1])
            ni = float(net_income[net_income.index.year == year].iloc[-1])
            if ic != 0:
                row['ROIC_%'] = ni / ic * 100
        if year in eps.index.year:
            row['EPS'] = float(eps[eps.index.year == year].iloc[-1])
        if np.isfinite(shares):
            row['Shares'] = float(shares)
        if year in fcf.index.year:
            row['FCF'] = float(fcf[fcf.index.year == year].iloc[-1])
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Year')
    return df


def run_diagnostics(finnhub_token, alphav_key, tickers=TEST_TICKERS):
    rows = []
    for symbol in tickers:
        result = {'ticker': symbol}
        try:
            profile = get_profile(symbol, finnhub_token)
            peers = get_peers(symbol, finnhub_token)
            info, hist = yahoo_price_history(symbol)
            fin = alphav_financials(symbol, alphav_key)
            base_fcf = fin['base_fcf']
            growth_base = fin['growth_base']
            shares = info.get('sharesOutstanding', np.nan)
            cash = info.get('totalCash', 0.0)
            debt = info.get('totalDebt', 0.0)
            pe_hist = historical_pe_series(fin['eps'], hist, max_years=10)
            base_path = intrinsic_path(base_fcf, growth_base, shares, cash, debt)
            result.update({
                'profile_ok': bool(profile.get('name') or profile.get('finnhubIndustry')),
                'peers_ok': len(peers) > 0,
                'financials_ok': len(fin['revenue']) > 0,
                'pe_hist_points': len(pe_hist),
                'valuation_ok': bool(np.isfinite(dcf_intrinsic_from_base(base_fcf, growth_base, shares, cash, debt))),
                'path_points': len(base_path),
                'status': 'OK'
            })
        except Exception as e:
            result.update({'status': f'ERROR: {e}'})
        rows.append(result)
    return pd.DataFrame(rows)

st.sidebar.header('Credenciales y parámetros')
finnhub_token = get_secret_or_input('FINNHUB_API_KEY', 'Finnhub API Key')
alpha_key = get_secret_or_input('ALPHAV_API_KEY', 'Alpha Vantage API Key')
ticker = st.sidebar.text_input('Ticker', 'AAPL').upper().strip()
custom_growth = st.sidebar.number_input('Crecimiento personalizado (%)', 0.0, 100.0, 10.0, 0.5) / 100.0
pess_cut = st.sidebar.slider('Recorte escenario pesimista (%)', 10, 90, 50) / 100.0
per_years = st.sidebar.slider('Años de PER histórico', 5, 15, 10)
run = st.sidebar.button('Analizar')
run_tests = st.sidebar.button('Ejecutar diagnóstico')

if run_tests:
    if not finnhub_token or not alpha_key:
        st.error('Necesitas API key de Finnhub y de Alpha Vantage para ejecutar el diagnóstico.')
    else:
        diag = run_diagnostics(finnhub_token, alpha_key)
        st.subheader('Diagnóstico automático')
        st.dataframe(diag, use_container_width=True)
        ok_count = int((diag['status'] == 'OK').sum()) if 'status' in diag.columns else 0
        st.caption(f'Tickers probados: {len(diag)} | Correctos: {ok_count}')

if run:
    if not finnhub_token or not alpha_key:
        st.error('Necesitas API keys de Finnhub y Alpha Vantage (FINNHUB_API_KEY y ALPHAV_API_KEY en secrets o barra lateral).')
    else:
        try:
            profile = get_profile(ticker, finnhub_token)
            peers = get_peers(ticker, finnhub_token)
            info, hist = yahoo_price_history(ticker)
            fin = alphav_financials(ticker, alpha_key)
            base_fcf = fin['base_fcf']
            g_base = fin['growth_base']
            g_pes = g_base * (1 - pess_cut)
            g_cus = custom_growth
            shares = info.get('sharesOutstanding', np.nan)
            cash = info.get('totalCash', 0.0)
            debt = info.get('totalDebt', 0.0)
            price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
            currency = info.get('currency', 'USD')
            name = profile.get('name') or info.get('longName') or ticker
            industry = profile.get('finnhubIndustry') or info.get('industry') or 'No disponible'
            country = profile.get('country') or 'No disponible'
            exchange = profile.get('exchange') or info.get('exchange') or 'No disponible'

            v_base = dcf_intrinsic_from_base(base_fcf, g_base, shares, cash, debt)
            v_pes = dcf_intrinsic_from_base(base_fcf, g_pes, shares, cash, debt)
            v_cus = dcf_intrinsic_from_base(base_fcf, g_cus, shares, cash, debt)
            pe_hist = historical_pe_series(fin['eps'], hist, max_years=per_years)
            growth_sales_pct = fin['growth_sales'] * 100 if np.isfinite(fin['growth_sales']) else None
            per_teorico = growth_sales_pct * 2 + 8 if growth_sales_pct is not None else None
            base_path = intrinsic_path(base_fcf, g_base, shares, cash, debt)
            pes_path = intrinsic_path(base_fcf, g_pes, shares, cash, debt)
            cus_path = intrinsic_path(base_fcf, g_cus, shares, cash, debt)
            mt_df = metrics_table_from_2020(fin, shares)

            st.subheader(name)
            st.write(f"Industria Finnhub: {industry} | País: {country} | Bolsa: {exchange}")
            st.caption(f"Peers detectados por Finnhub: {', '.join(peers) if peers else 'No disponibles'}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Precio actual', f"{price:.2f} {currency}" if np.isfinite(price) else 'No disp.')
            c2.metric('Valor base', f"{v_base:.2f} {currency}" if np.isfinite(v_base) else 'No disp.')
            c3.metric('Valor pesimista', f"{v_pes:.2f} {currency}" if np.isfinite(v_pes) else 'No disp.')
            c4.metric('Valor personalizado', f"{v_cus:.2f} {currency}" if np.isfinite(v_cus) else 'No disp.')

            st.subheader('PER histórico real (Alpha Vantage + precios Yahoo)')
            fig1 = go.Figure()
            if len(pe_hist):
                fig1.add_trace(go.Scatter(x=pe_hist.index.astype(str), y=pe_hist.values, mode='lines+markers', name='PER empresa'))
            fig1.update_layout(height=360, xaxis_title='Año', yaxis_title='PER')
            st.plotly_chart(fig1, use_container_width=True)
            forward_pe = info.get('forwardPE', np.nan)
            forward_pe_str = f"{forward_pe:.2f}" if np.isfinite(forward_pe) else 'No disponible'
            if per_teorico is not None:
                st.caption(f"Forward PER actual: {forward_pe_str} | PER teórico (2× crecimiento ventas + 8): {per_teorico:.2f}")
            else:
                st.caption(f"Forward PER actual: {forward_pe_str} | PER teórico: No disponible")

            st.subheader('Precio vs valor relativo a 5 años')
            fig3 = go.Figure()
            years = list(base_path.index)
            fig3.add_trace(go.Scatter(x=years, y=[price] * len(years), mode='lines+markers', name='Precio actual'))
            fig3.add_trace(go.Scatter(x=years, y=base_path.values, mode='lines+markers', name='Valor base'))
            fig3.add_trace(go.Scatter(x=years, y=pes_path.values, mode='lines+markers', name='Valor pesimista'))
            fig3.add_trace(go.Scatter(x=years, y=cus_path.values, mode='lines+markers', name='Valor personalizado'))
            fig3.update_layout(height=380, xaxis_title='Años desde hoy', yaxis_title=f'Valor ({currency})')
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader('Métricas clave desde 2020 (Alpha Vantage)')
            if not mt_df.empty:
                display_df = mt_df.copy()
                for col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: round(x, 2) if pd.notna(x) else x)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.caption('No hay datos suficientes para construir la tabla desde 2020.')

            st.subheader('Arquitectura de datos')
            st.markdown('- Perfil y peers: **Finnhub**.')
            st.markdown('- Estados financieros históricos (income, balance, cash flow, EPS): **Alpha Vantage**.')
            st.markdown('- Precios históricos para PER y valoraciones: **Yahoo Finance / yfinance** con caché.')
            st.markdown(f'- Base local: **SQLite ({DB_PATH})** para perfiles y peers.')
        except Exception as e:
            st.error(f'Error al analizar {ticker}: {e}')
