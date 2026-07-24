import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Calculadora Value Investing", layout="wide")

st.title("📈 Valoración de Acciones - Modelo Warren Buffett")
st.markdown("""
Esta herramienta evalúa la calidad del negocio (ROIC, Márgenes) y calcula el valor intrínseco 
descontando los flujos de caja libre a 10 años (WACC 7.5%, Tasa Terminal 3%).
""")

# Barra lateral para Inputs
st.sidebar.header("Parámetros de Análisis")
ticker = st.sidebar.text_input("Ticker de la Empresa (Ej. AAPL, MSFT, ITX.MC):", "AAPL").upper()
custom_growth = st.sidebar.number_input("Crecimiento Personalizado (Anual %):", min_value=0.0, max_value=100.0, value=10.0) / 100

if st.sidebar.button("Analizar Empresa"):
    with st.spinner(f"Extrayendo datos financieros de {ticker}..."):
        try:
            # 1. Extracción de Datos
            stock = yf.Ticker(ticker)
            info = stock.info
            inc = stock.financials
            cf = stock.cashflow
            bs = stock.balance_sheet
            
            precio = info.get('currentPrice', info.get('regularMarketPrice', 0))
            shares = info.get('sharesOutstanding', 1)
            divisa = info.get('currency', 'USD')
            
            if precio == 0 or inc.empty or cf.empty or bs.empty:
                st.error(f"No se han podido obtener datos completos para {ticker}. Verifica el Ticker.")
                st.stop()

            # 2. Extracción para Métricas de Calidad
            # ROIC
            ebit = inc.loc['EBIT'].iloc[0] if 'EBIT' in inc.index else inc.loc['Operating Income'].iloc[0]
            pretax = inc.loc['Pretax Income'].iloc[0] if 'Pretax Income' in inc.index else ebit
            tax_prov = inc.loc['Tax Provision'].iloc[0] if 'Tax Provision' in inc.index else pretax * 0.21
            tax_rate = tax_prov / pretax if pretax > 0 else 0.21
            nopat = ebit * (1 - tax_rate)
            
            total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
            equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else bs.loc['Total Equity Gross Minority Interest'].iloc[0]
            roic = (nopat / (total_debt + equity)) * 100 if (total_debt + equity) > 0 else 0

            # Crecimiento de Ventas Histórico
            rev = inc.loc['Total Revenue'].dropna().sort_index(ascending=True) if 'Total Revenue' in inc.index else inc.loc['Operating Revenue'].dropna().sort_index(ascending=True)
            crec_ventas = rev.pct_change().mean() * 100 if len(rev) > 1 else 0

            # Margen EBITDA
            ebitda = inc.loc['EBITDA'].iloc[0] if 'EBITDA' in inc.index else ebit * 1.15
            margen_ebitda = (ebitda / rev.iloc[-1]) * 100 if rev.iloc[-1] > 0 else 0

            # 3. Datos para el Descuento de Flujos (DCF)
            ocf = cf.loc['Operating Cash Flow'].dropna().sort_index(ascending=True)
            capex = cf.loc['Capital Expenditure'].dropna().sort_index(ascending=True)
            capex = -np.abs(capex) # Asegurar que es negativo
            
            fcf = ocf + capex
            base_fcf = np.mean(fcf.iloc[-2:]) if len(fcf) >= 2 else fcf.iloc[-1]

            # Crecimiento de Beneficios (Net Income)
            ni = inc.loc['Net Income'].dropna().sort_index(ascending=True)
            crec_real_ni = ni.pct_change().mean() if len(ni) >= 2 else 0.05
            if pd.isna(crec_real_ni) or np.isinf(crec_real_ni): crec_real_ni = 0.05

            # Escenarios de Crecimiento
            crec_base = min(max(crec_real_ni, 0.04), 0.20) # Topado al 20%, Suelo 4%
            crec_pesimista = crec_base / 2
            
            # Caja Total
            cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
            sti = bs.loc['Other Short Term Investments'].iloc[0] if 'Other Short Term Investments' in bs.index else 0
            total_cash = cash + sti

            # 4. Motor DCF
            def calcular_dcf(growth_rate):
                if base_fcf <= 0: return 0
                wacc = 0.075
                term_growth = 0.03
                
                fcf_proy = []
                curr_fcf = base_fcf
                for _ in range(5):
                    curr_fcf *= (1 + growth_rate)
                    fcf_proy.append(curr_fcf)
                slowdown = growth_rate
                for _ in range(5):
                    slowdown = max(slowdown * 0.8, 0.03)
                    curr_fcf *= (1 + slowdown)
                    fcf_proy.append(curr_fcf)
                    
                pv_fcf = sum([f / ((1+wacc)**(i+1)) for i, f in enumerate(fcf_proy)])
                vt = (fcf_proy[-1] * (1+term_growth)) / (wacc - term_growth)
                pv_vt = vt / ((1+wacc)**10)
                
                ev = pv_fcf + pv_vt
                equity_val = ev + total_cash - total_debt
                return equity_val / shares if shares > 0 else 0

            val_base = calcular_dcf(crec_base)
            val_pesimista = calcular_dcf(crec_pesimista)
            val_custom = calcular_dcf(custom_growth)

            # --- RENDERIZADO DEL DASHBOARD ---
            st.header(f"📊 {info.get('longName', ticker)}")
            st.metric("Precio Actual", f"{precio} {divisa}")
            
            st.subheader("1. Métricas de Calidad (Filtro de Buffett)")
            col1, col2, col3 = st.columns(3)
            col1.metric("ROIC (Retorno Capital)", f"{roic:.2f}%", "Ideal > 15%")
            col2.metric("Margen EBITDA", f"{margen_ebitda:.2f}%")
            col3.metric("Crecimiento Histórico Ventas", f"{crec_ventas:.2f}%")

            st.subheader("2. Valoración DCF y Márgenes de Seguridad")
            if base_fcf <= 0:
                st.warning("⚠️ La empresa tiene Flujo de Caja Libre negativo (o es intensiva en capital/financiera). El modelo DCF no es aplicable. Valórala por Precio/Valor en Libros.")
            else:
                col4, col5, col6 = st.columns(3)
                
                # BASE
                mos_base = ((val_base - precio) / val_base) * 100 if val_base > 0 else 0
                col4.info(f"**ESCENARIO BASE**\n\nCrecimiento: {crec_base*100:.2f}%")
                col4.metric("Valor Intrínseco", f"{val_base:.2f} {divisa}")
                col4.metric("Margen de Seguridad", f"{mos_base:.2f}%", delta_color="normal" if mos_base > 0 else "inverse")

                # PESIMISTA
                mos_pes = ((val_pesimista - precio) / val_pesimista) * 100 if val_pesimista > 0 else 0
                col5.error(f"**ESCENARIO PESIMISTA**\n\nCrecimiento: {crec_pesimista*100:.2f}%")
                col5.metric("Valor Intrínseco", f"{val_pesimista:.2f} {divisa}")
                col5.metric("Margen de Seguridad", f"{mos_pes:.2f}%", delta_color="normal" if mos_pes > 0 else "inverse")

                # PERSONALIZADO
                mos_cust = ((val_custom - precio) / val_custom) * 100 if val_custom > 0 else 0
                col6.success(f"**ESCENARIO PERSONALIZADO**\n\nCrecimiento: {custom_growth*100:.2f}%")
                col6.metric("Valor Intrínseco", f"{val_custom:.2f} {divisa}")
                col6.metric("Margen de Seguridad", f"{mos_cust:.2f}%", delta_color="normal" if mos_cust > 0 else "inverse")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar los datos: {e}")