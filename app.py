import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from bcb import sgs
from bcb import Expectativas
import pyettj
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.optimize import minimize
import io
import base64

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILIZAÇÃO ---
st.set_page_config(page_title="Ghia MFO - Asset Allocation", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 200px; max-width: 250px; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    
    /* Radio buttons na página - item selecionado em branco */
    div[role="radiogroup"] > label > div:first-of-type { display: none; }
    div[role="radiogroup"] label { padding-left: 0px !important; }
    div[role="radiogroup"] p { font-size: 15px; font-weight: 500; color: #808495; }
    div[role="radiogroup"] label[data-baseweb="radio"] input:checked ~ div p { color: #FFFFFF !important; font-weight: 700; }
    
    /* Selectbox (perfil) - texto selecionado em branco */
    div[data-baseweb="select"] span { color: #FFFFFF !important; font-weight: 600; }
    
    /* Menu lateral - páginas */
    [data-testid="stSidebar"] button[kind="secondary"] p { color: #808495 !important; }
    [data-testid="stSidebar"] button[kind="primary"] p { color: #FFFFFF !important; font-weight: 700 !important; }
    
    header {visibility: hidden;}
    .stAlert { padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNÇÕES DE DADOS PÚBLICOS ---

@st.cache_data
def get_historico_real():
    try:
        df = sgs.get({'IPCA': 433, 'CDI': 4391}, start='2005-01-01')
        df = df / 100
        df['Juro_Real'] = ((1 + df['CDI']) / (1 + df['IPCA'])) - 1
        df['Juro_Real_12m'] = df['Juro_Real'].rolling(12).apply(lambda x: np.prod(1 + x) - 1)
        df = df.dropna()
        if len(df) >= 60:
            media_5y = df['Juro_Real_12m'].iloc[-60:].mean()
        else:
            media_5y = df['Juro_Real_12m'].mean()
        return df, media_5y
    except Exception:
        return pd.DataFrame(), 0.05

@st.cache_data
def get_expectativa_inflacao_12m():
    try:
        em = Expectativas()
        ep = em.get_endpoint('ExpectativasMercadoInflacao12Meses')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df_focus = ep.query().filter(ep.Indicador == 'IPCA').filter(ep.baseCalculo == 0).filter(ep.Data >= start_date).collect()
        if not df_focus.empty:
            df_focus = df_focus.sort_values('Data')
            return df_focus.iloc[-1]['Mediana'] / 100
        return 0.04
    except Exception:
        return 0.04

@st.cache_data
def get_ettj_1ano():
    for i in range(0, 5):
        data_busca = datetime.now() - timedelta(days=i)
        data_str = data_busca.strftime("%d/%m/%Y")
        try:
            curva = pyettj.get_ettj(data_str)
            if not curva.empty and 'Dias Corridos' in curva.columns:
                curva = curva.sort_values('Dias Corridos')
                f_interp = interpolate.interp1d(curva['Dias Corridos'], curva['DI x pré 252'], kind='linear', fill_value="extrapolate")
                return float(f_interp(365)) / 100, data_str
        except Exception:
            continue
    return 0.12, "Erro/Fallback"

@st.cache_data
def get_taxa_pre_vertice_504():
    """Busca taxa pré do vértice 504 dias da ETTJ"""
    for i in range(0, 10):  # Aumenta busca para 10 dias (pega semana anterior)
        data_busca = datetime.now() - timedelta(days=i)
        # Pula finais de semana
        if data_busca.weekday() >= 5:  # 5=sábado, 6=domingo
            continue
        
        data_str = data_busca.strftime("%d/%m/%Y")
        try:
            # USA get_ettj_anbima() para obter vértices padronizados da ANBIMA
            # Retorna tupla: (params_svensson, vertices_anbima, prefixados, titulos)
            resultado = pyettj.get_ettj_anbima(data_str)
            curva = resultado[1]  # Pega o DataFrame de vértices (índice 1)
            
            # Colunas: ['Vertice', 'IPCA', 'Prefixados', 'Inflação Implícita']
            if not curva.empty and 'Vertice' in curva.columns and 'Prefixados' in curva.columns:
                # Converte colunas para numérico (trata formatação brasileira)
                # Remove pontos de milhar ANTES de converter (ex: "1.008" → "1008")
                curva['Vertice'] = curva['Vertice'].apply(lambda x: str(x).replace('.', ''))
                curva['Vertice'] = pd.to_numeric(curva['Vertice'], errors='coerce')
                # Converte taxas: troca vírgula por ponto (ex: "13,0208" → 13.0208)
                curva['Prefixados'] = curva['Prefixados'].apply(
                    lambda x: float(str(x).replace(',', '.')) if pd.notna(x) and str(x).strip() != '' else None
                )
                # Remove linhas com valores vazios
                curva = curva.dropna(subset=['Vertice', 'Prefixados'])
                curva = curva.sort_values('Vertice')
                
                # Com get_ettj_anbima(), o vértice 504 deve existir exatamente
                if 504 in curva['Vertice'].values:
                    taxa = curva[curva['Vertice'] == 504]['Prefixados'].iloc[0]
                    valor_interpolado = False
                else:
                    # Fallback: interpola se não existir (não deveria acontecer)
                    f_interp = interpolate.interp1d(curva['Vertice'], curva['Prefixados'], 
                                                   kind='linear', fill_value="extrapolate")
                    taxa = float(f_interp(504))
                    valor_interpolado = True
                
                # Debug: salva info da curva
                vertice_504_row = curva[curva['Vertice'] == 504] if 504 in curva['Vertice'].values else None
                st.session_state['debug_ettj_pre'] = {
                    'data': data_str,
                    'colunas': curva.columns.tolist(),
                    'sample_completo': curva.to_dict('records'),
                    'vertice_504_existe': (504 in curva['Vertice'].values),
                    'taxa_retornada': taxa,
                    'foi_interpolado': valor_interpolado,
                    'vertice_504_row': vertice_504_row.to_dict('records') if vertice_504_row is not None and not vertice_504_row.empty else None,
                    'min_dias': curva['Vertice'].min(),
                    'max_dias': curva['Vertice'].max()
                }
                
                return taxa / 100, data_str  # Já vem em %
        except Exception as e:
            st.session_state['debug_ettj_pre_error'] = str(e)
            continue
    return 0.12, "Erro/Fallback"

@st.cache_data
def get_taxa_real_vertice_1638():
    """Busca taxa real (DI x IPCA) do vértice 1638 dias da ETTJ"""
    for i in range(0, 10):  # Aumenta busca para 10 dias (pega semana anterior)
        data_busca = datetime.now() - timedelta(days=i)
        # Pula finais de semana
        if data_busca.weekday() >= 5:  # 5=sábado, 6=domingo
            continue
        
        data_str = data_busca.strftime("%d/%m/%Y")
        try:
            # USA get_ettj_anbima() para obter vértices padronizados da ANBIMA
            # Retorna tupla: (params_svensson, vertices_anbima, prefixados, titulos)
            resultado = pyettj.get_ettj_anbima(data_str)
            curva = resultado[1]  # Pega o DataFrame de vértices (índice 1)
            
            # Colunas: ['Vertice', 'IPCA', 'Prefixados', 'Inflação Implícita']
            if not curva.empty and 'Vertice' in curva.columns and 'IPCA' in curva.columns:
                # Converte colunas para numérico (trata formatação brasileira)
                # Remove pontos de milhar ANTES de converter (ex: "1.638" → "1638")
                curva['Vertice'] = curva['Vertice'].apply(lambda x: str(x).replace('.', ''))
                curva['Vertice'] = pd.to_numeric(curva['Vertice'], errors='coerce')
                # Converte taxas: troca vírgula por ponto (ex: "7,4832" → 7.4832)
                curva['IPCA'] = curva['IPCA'].apply(
                    lambda x: float(str(x).replace(',', '.')) if pd.notna(x) and str(x).strip() != '' else None
                )
                # Remove linhas com valores vazios
                curva = curva.dropna(subset=['Vertice', 'IPCA'])
                curva = curva.sort_values('Vertice')
                
                # Com get_ettj_anbima(), o vértice 1638 deve existir exatamente
                if 1638 in curva['Vertice'].values:
                    taxa = curva[curva['Vertice'] == 1638]['IPCA'].iloc[0]
                    valor_interpolado = False
                else:
                    # Fallback: interpola se não existir (não deveria acontecer)
                    f_interp = interpolate.interp1d(curva['Vertice'], curva['IPCA'], 
                                                   kind='linear', fill_value="extrapolate")
                    taxa = float(f_interp(1638))
                    valor_interpolado = True
                
                # Debug: salva info da curva
                vertice_1638_row = curva[curva['Vertice'] == 1638] if 1638 in curva['Vertice'].values else None
                st.session_state['debug_ettj_real'] = {
                    'data': data_str,
                    'colunas': curva.columns.tolist(),
                    'sample_completo': curva.to_dict('records'),
                    'vertice_1638_existe': (1638 in curva['Vertice'].values),
                    'taxa_retornada': taxa,
                    'foi_interpolado': valor_interpolado,
                    'vertice_1638_row': vertice_1638_row.to_dict('records') if vertice_1638_row is not None and not vertice_1638_row.empty else None,
                    'min_dias': curva['Vertice'].min(),
                    'max_dias': curva['Vertice'].max()
                }
                
                return taxa / 100, data_str  # Já vem em %
        except Exception as e:
            st.session_state['debug_ettj_real_error'] = str(e)
            continue
    return 0.064, "Erro/Fallback"

@st.cache_data
def get_taxa_real_ex_ante_252():
    """Busca taxa real ex-ante (DI x IPCA) do vértice 252 dias úteis (1 ano) da ETTJ ANBIMA"""
    for i in range(0, 10):
        data_busca = datetime.now() - timedelta(days=i)
        # Pula finais de semana
        if data_busca.weekday() >= 5:
            continue
        
        data_str = data_busca.strftime("%d/%m/%Y")
        try:
            resultado = pyettj.get_ettj_anbima(data_str)
            curva = resultado[1]  # DataFrame de vértices
            
            if not curva.empty and 'Vertice' in curva.columns and 'IPCA' in curva.columns:
                # Converte colunas para numérico
                curva['Vertice'] = curva['Vertice'].apply(lambda x: str(x).replace('.', ''))
                curva['Vertice'] = pd.to_numeric(curva['Vertice'], errors='coerce')
                curva['IPCA'] = curva['IPCA'].apply(
                    lambda x: float(str(x).replace(',', '.')) if pd.notna(x) and str(x).strip() != '' else None
                )
                curva = curva.dropna(subset=['Vertice', 'IPCA'])
                curva = curva.sort_values('Vertice')
                
                # Busca vértice 252 dias úteis (1 ano)
                if 252 in curva['Vertice'].values:
                    taxa = curva[curva['Vertice'] == 252]['IPCA'].iloc[0]
                    valor_interpolado = False
                else:
                    # Interpola se não existir exato
                    f_interp = interpolate.interp1d(curva['Vertice'], curva['IPCA'], 
                                                   kind='linear', fill_value="extrapolate")
                    taxa = float(f_interp(252))
                    valor_interpolado = True
                
                # Debug
                st.session_state['debug_ettj_real_exante'] = {
                    'data': data_str,
                    'vertice_252_existe': (252 in curva['Vertice'].values),
                    'taxa_retornada': taxa,
                    'foi_interpolado': valor_interpolado
                }
                
                return taxa / 100, data_str
        except Exception as e:
            st.session_state['debug_ettj_real_exante_error'] = str(e)
            continue
    return 0.05, "Erro/Fallback"

# --- 3. FUNÇÕES DE OTIMIZAÇÃO ---

def black_litterman(S, market_caps, tau, P, Q, omega):
    """
    Implementação do modelo Black-Litterman
    
    Args:
        S: Matriz de covariância (numpy array)
        market_caps: Pesos de equilíbrio de mercado (numpy array)
        tau: Parâmetro de incerteza (scalar, tipicamente entre 0.01 e 0.05)
        P: Matriz de views (numpy array) - cada linha é uma view
        Q: Vetor de retornos esperados das views (numpy array)
        omega: Matriz de incerteza das views (numpy array, diagonal)
    
    Returns:
        mu_bl: Retornos esperados ajustados pelo Black-Litterman
    """
    # Calcula o vetor de retornos implícitos do mercado (Pi)
    # Assume risk aversion = 2.5 (típico)
    risk_aversion = 2.5
    pi = risk_aversion * S @ market_caps
    
    # Fórmula do Black-Litterman
    # mu_BL = [(tau*S)^-1 + P'*Omega^-1*P]^-1 * [(tau*S)^-1*Pi + P'*Omega^-1*Q]
    tau_S = tau * S
    tau_S_inv = np.linalg.inv(tau_S)
    
    omega_inv = np.linalg.inv(omega)
    
    # Termo da esquerda: [(tau*S)^-1 + P'*Omega^-1*P]^-1
    left_term = np.linalg.inv(tau_S_inv + P.T @ omega_inv @ P)
    
    # Termo da direita: [(tau*S)^-1*Pi + P'*Omega^-1*Q]
    right_term = tau_S_inv @ pi + P.T @ omega_inv @ Q
    
    # Retorno esperado ajustado
    mu_bl = left_term @ right_term
    
    return mu_bl

def risk_parity_optimization(S, vol_target=None):
    """
    Otimização Risk Parity - equaliza contribuição de risco
    
    Args:
        S: Matriz de covariância (numpy array)
        vol_target: Volatilidade alvo anual (opcional). Se None, retorna pesos não alavancados
    
    Returns:
        weights: Pesos otimizados (numpy array)
    """
    num_assets = S.shape[0]
    
    # Função objetivo: minimizar a diferença entre contribuições de risco
    def risk_contribution_diff(w):
        """Calcula diferença entre contribuições de risco (queremos minimizar)"""
        portfolio_var = w.T @ S @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Contribuição marginal de risco
        marginal_contrib = S @ w
        
        # Contribuição de risco de cada ativo
        risk_contrib = w * marginal_contrib / portfolio_vol if portfolio_vol > 0 else w * 0
        
        # Contribuição de risco alvo (igual para todos)
        target_contrib = portfolio_var / num_assets
        
        # Soma dos quadrados das diferenças
        return np.sum((risk_contrib - target_contrib)**2)
    
    # Chute inicial: pesos iguais
    init_guess = num_assets * [1. / num_assets,]
    
    # Restrições: soma = 1, todos positivos
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Otimização
    result = minimize(
        risk_contribution_diff,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        weights = result.x
        
        # Se houver target de volatilidade, aplica alavancagem
        if vol_target is not None:
            current_vol = np.sqrt(weights.T @ S @ weights)
            leverage = vol_target / current_vol if current_vol > 0 else 1
            weights = weights * leverage
        
        return weights
    else:
        # Fallback: pesos iguais
        return np.array(init_guess)

# --- 4. INTEGRAÇÃO COMDINHEIRO (HARDCODED PAYLOAD) ---
def fetch_comdinheiro_data(user, password):
    """
    Usa a string exata fornecida pelo suporte da Comdinheiro, apenas injetando o login.
    Mudei format=json3 para json2 para facilitar o processamento tabular.
    """
    url = "https://api.comdinheiro.com.br/v1/ep1/import-data"
    
    # Payload com JSON2 e max_list_size=10000
    # REMOVIDO cabecalho_excel=modo1 que pode estar causando retorno apenas de metadados
    payload = (
        f"username={user}&password={password}&"
        "URL=HistoricoCotacao002.php%3F%26x%3Dcdi%2Bibov%2Bifix%2Banbima_imab%2Banbima_irfm%2Banbima_idadi%2Banbima_ihfa%2Banbima_imab5%25BE%26data_ini%3D04012013%26data_fim%3D05122025%26pagina%3D1%26d%3DMOEDA_ORIGINAL%26g%3D1%26m%3D0%26info_desejada%3Dretorno%26retorno%3Ddiscreto%26tipo_data%3Ddu_br%26tipo_ajuste%3Dtodosajustes%26num_casas%3D2%26enviar_email%3D0%26ordem_legenda%3D1%26classes_ativos%3Dfklk448oj5v5r%26ordem_data%3D0%26rent_acum%3Drent_acum%26minY%3D%26maxY%3D%26deltaY%3D%26preco_nd_ant%3D0%26base_num_indice%3D100%26flag_num_indice%3D0%26eixo_x%3DData%26startX%3D0%26max_list_size%3D10000%26line_width%3D2%26titulo_grafico%3D%26legenda_eixoy%3D%26tipo_grafico%3Dline%26script%3D%26tooltip%3Dunica&format=json2"
    )
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, dict) and 'erro' in data:
                return None, f"Erro API: {data['erro']}"
            
# Parser JSON2
            if 'resposta' in data:
                # Verifica se há dados em tab-p1 ao invés de tab-p0
                rows_p0 = data['resposta'].get('tab-p0', {}).get('linha', []) if 'tab-p0' in data['resposta'] else []
                rows_p1 = data['resposta'].get('tab-p1', {}).get('linha', []) if 'tab-p1' in data['resposta'] else []
                
                # DEBUG COMPLETO DA RESPOSTA API
                st.session_state['api_full_response'] = {
                    'total_rows_p0': len(rows_p0),
                    'total_rows_p1': len(rows_p1),
                    'raw_json_keys': list(data.keys()),
                    'resposta_keys': list(data['resposta'].keys()) if 'resposta' in data else [],
                    'rows_p0': rows_p0[:10] if len(rows_p0) > 10 else rows_p0,
                    'rows_p1': rows_p1[:10] if len(rows_p1) > 10 else rows_p1,
                    'all_rows_p0': rows_p0,
                    'all_rows_p1': rows_p1
                }
                
                # Tenta usar tab-p1 se tab-p0 só tiver metadados
                if len(rows_p1) > len(rows_p0):
                    rows = rows_p1
                else:
                    rows = rows_p0
                
                if not rows:
                    return None, f"Nenhuma linha em tab-p0 ou tab-p1. Keys: {list(data.keys())}"
                df = pd.DataFrame(rows)
                
                # DEBUG: Salvar dados brutos em session_state para visualização
                st.session_state['api_raw_data'] = {
                    'df_original': df.copy(),
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'first_rows': df.head(10).to_dict('records'),
                    'all_rows': df.to_dict('records'),  # TODAS as linhas
                    'dtypes': df.dtypes.to_dict()
                }
                
                # Verifica qual coluna de data existe (pode ser 'descricao' ou 'data')
                if 'descricao' in df.columns:
                    # TAB-P0: tem 'descricao' mas são só metadados
                    # PULA AS PRIMEIRAS 3 LINHAS QUE SÃO METADADOS
                    df = df.iloc[3:].copy()
                    if len(df) == 0:
                        return None, "TAB-P0 só tem metadados, nenhum dado real."
                    df = df.rename(columns={'descricao': 'data'})
                elif 'data' in df.columns:
                    # TAB-P1: já vem com 'data' e são dados reais (sem metadados)
                    pass
                else:
                    return None, f"Coluna de data não encontrada. Colunas: {list(df.columns)}"
                
                # Tratamento de Data
                df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y", errors='coerce')
                df = df.dropna(subset=['data'])  # Remove linhas sem data válida
                df = df.set_index('data')
                
                # Mapeamento das colunas que vêm da API para os nomes do GHIA
                mapa = {
                    "cdi": "CDI (Caixa)",
                    "anbima_idadi": "IDA-DI (Crédito)",
                    "anbima_imab": "IMA-B (Inflação)",
                    "anbima_imab5+": "IMA-B 5+ (Ilíquidos)",
                    "anbima_irfm": "IRF-M (Pré-fixado)",
                    "anbima_ihfa": "IHFA (Multimercado)",
                    "ibov": "IBOV (Ações BR)",
                    "ifix": "IFIX (FIIs)"
                }
                
                # Converte valores para numérico (já vêm em formato decimal, NÃO dividir por 100!)
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in mapa.keys():
                        # Converter números (PT-BR: vírgula para ponto)
                        # Remove "BRL" e outros textos que possam estar misturados
                        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('BRL', '').str.strip()
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # A API já retorna em formato DECIMAL (0.0006 = 0.06% retorno)
                        # NÃO dividir por 100 novamente!
                
                # Renomeia colunas que existirem (case-insensitive)
                cols_to_rename = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in mapa:
                        cols_to_rename[col] = mapa[col_lower]
                
                df = df.rename(columns=cols_to_rename)
                
                # Mantém apenas as colunas mapeadas
                cols_finais = [v for v in mapa.values() if v in df.columns]
                if not cols_finais:
                    return None, f"Nenhuma coluna de ativo reconhecida após mapeamento. Colunas processadas: {df.columns.tolist()}"
                
                df = df[cols_finais]
                
                # Remove dias sem nenhum dado válido (todas colunas NaN)
                df = df.dropna(how='all')
                
                # Para covariância, melhor dropar NaNs do que preencher com 0
                # 0 representa retorno zero, não ausência de dado
                df = df.dropna()
                
                if len(df) == 0:
                    return None, "DataFrame vazio após limpeza. Verifique os dados da API."
                
                return df, f"Sucesso - {len(df)} dias de dados"
            else:
                return None, "Formato JSON inesperado."
        else:
            return None, f"Status HTTP: {response.status_code}"
            
    except Exception as e:
        return None, str(e)

# --- 4. EXECUÇÃO GLOBAL ---
with st.spinner('Sincronizando dados macroeconômicos...'):
    df_hist, media_juro_real_5y = get_historico_real()
    taxa_di_1y, data_ettj = get_ettj_1ano()
    expectativa_ipca = get_expectativa_inflacao_12m()
    
    # Busca taxa real ex-ante diretamente da ETTJ IPCA 252 dias
    taxa_real_ex_ante, data_real_exante = get_taxa_real_ex_ante_252()
    
    # Busca taxas dos vértices específicos da ETTJ
    taxa_pre_504, data_pre_504 = get_taxa_pre_vertice_504()
    taxa_real_1638, data_real_1638 = get_taxa_real_vertice_1638()

    if 'taxa_di_1y' not in st.session_state: st.session_state['taxa_di_1y'] = taxa_di_1y
    if 'data_ettj' not in st.session_state: st.session_state['data_ettj'] = data_ettj
    if 'taxa_real_ex_ante' not in st.session_state: st.session_state['taxa_real_ex_ante'] = taxa_real_ex_ante
    if 'data_real_exante' not in st.session_state: st.session_state['data_real_exante'] = data_real_exante
    if 'taxa_pre_504' not in st.session_state: st.session_state['taxa_pre_504'] = taxa_pre_504
    if 'data_pre_504' not in st.session_state: st.session_state['data_pre_504'] = data_pre_504
    if 'taxa_real_1638' not in st.session_state: st.session_state['taxa_real_1638'] = taxa_real_1638
    if 'data_real_1638' not in st.session_state: st.session_state['data_real_1638'] = data_real_1638

# --- 5. NAVEGAÇÃO ---
st.sidebar.markdown("### Navegação")
pagina = st.sidebar.radio("Ir para:", ["Benchmark", "Cenários Macro", "Otimização"], label_visibility="collapsed")
st.sidebar.markdown("---")
with st.sidebar.container():
    st.markdown(f"""
    <div style="background-color: #262730; padding: 10px; border-radius: 5px; font-size: 12px; color: #DDD;">
        <strong style="color: #189CD8">Dados de Mercado</strong><br>
        Base Curva: {data_ettj}<br>
        DI Futuro (1Y): {taxa_di_1y*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

if pagina == "Otimização":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔐 Acesso Comdinheiro")
    api_user = st.sidebar.text_input("Usuário", value=st.session_state.get('api_user', 'InteligenciaGhia'))
    api_pass = st.sidebar.text_input("Senha", type="password", value=st.session_state.get('api_pass', 'InteligenciaGhia'))
    st.session_state['api_user'] = api_user
    st.session_state['api_pass'] = api_pass

# --- 6. CONTEÚDO ---

# === PÁGINA 1: BENCHMARK ===
if pagina == "Benchmark":
    st.title("Definição de Benchmark de Rentabilidade")
    
    st.markdown("""
    Esta etapa estabelece o **benchmark com base no juro real** que servirá de base para as metas das carteiras.
    O cálculo combina o histórico (juro real Ex-Post) com as expectativas atuais de mercado (juro real Ex-Ante).
    """)
    
    st.divider()

    # --- Bloco A: Diagnóstico ---
    st.subheader("1. Diagnóstico de Juro Real")
    
    if not df_hist.empty:
        media_total_ex_post = df_hist['Juro_Real_12m'].mean()
        media_5y_ex_post = df_hist['Juro_Real_12m'].iloc[-60:].mean()
        atual_ex_post = df_hist['Juro_Real_12m'].iloc[-1]
        data_ultimo = df_hist.index[-1].strftime('%m/%Y')
    else:
        media_total_ex_post = 0; media_5y_ex_post = 0; atual_ex_post = 0; data_ultimo = "-"

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.info("**Visão Histórica (Ex-Post)**\n\nRetorno acumulado de 12 meses do CDI descontado pelo IPCA realizado no mesmo período.")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Média Desde 2005", 
            f"{media_total_ex_post*100:.2f}%",
            help="Média aritmética simples do juro real ex-post de todos os meses desde janeiro/2005. Representa o retorno médio histórico do CDI acima da inflação em períodos normais e de crise."
        )
        c2.metric(
            "Média 5 Anos", 
            f"{media_5y_ex_post*100:.2f}%", 
            help="Média móvel dos últimos 60 meses (5 anos). Captura tendências mais recentes e é menos influenciada por períodos antigos de juro real elevado (2015-2016). Recomendada para benchmarks de médio prazo."
        )
        c3.metric(
            f"Atual ({data_ultimo})", 
            f"{atual_ex_post*100:.2f}%", 
            delta=f"{(atual_ex_post - media_5y_ex_post)*100:.2f} p.p vs média",
            help=f"Juro real realizado nos últimos 12 meses (acumulado móvel até {data_ultimo}). Calculado como: [(1+CDI acum)/(1+IPCA acum)] - 1. O delta mostra quantos pontos percentuais está acima/abaixo da média de 5 anos."
        )

    with col_right:
        st.warning(f"**Visão Mercado (Ex-Ante)**\n\nExpectativa implícita na Curva de Juros ANBIMA e IPCA esperado pelo mercado.")
        c4, c5 = st.columns(2)
        c4.metric(
            "DI Futuro (1Y)", 
            f"{taxa_di_1y*100:.2f}%", 
            help=f"Taxa pré-fixada extraída da Estrutura a Termo da Taxa de Juros (ETTJ) da ANBIMA para o vértice de 365 dias corridos. Representa a expectativa do mercado para o CDI médio nos próximos 12 meses. Data: {data_ettj}"
        )
        c5.metric(
            "IPCA Focus (12m)", 
            f"{expectativa_ipca*100:.2f}%", 
            help="Mediana das expectativas de inflação para os próximos 12 meses coletadas pelo Banco Central no Boletim Focus. Atualizado semanalmente com as projeções de cerca de 100 instituições financeiras."
        )
        st.metric(
            "Juro Real Ex-Ante (1Y)", 
            f"{taxa_real_ex_ante*100:.2f}%", 
            delta="Taxa de Mercado",
            help=f"Taxa real (DI x IPCA) implícita na curva ANBIMA para o vértice de 252 dias úteis (1 ano). Extraída diretamente da ETTJ IPCA, reflete a expectativa do mercado de juro real para os próximos 12 meses. Data: {data_real_exante}"
        )

    st.divider()

    # --- Bloco B: Definição do Target ---
    st.subheader("2. Definição do Target")
    st.markdown("""
    Defina o peso atribuído ao histórico versus a expectativa de mercado para compor o **Benchmark Híbrido**.
    Carteiras de longo prazo tendem a atribuir maior peso à média histórica para evitar volatilidade de meta.
    """)
    
    # Carrega configurações salvas ou usa defaults
    default_benchmark_config = {
        'periodo_historico': '5 Anos',
        'peso_hist': 50.0,
        'override': False,
        'valor_manual': 6.0,
        'usar_di_futuro': False
    }
    saved_benchmark_config = st.session_state.get('benchmark_config_salvo', default_benchmark_config)
    
    # Opção de usar DI Futuro 1Y como benchmark
    usar_di_futuro = st.checkbox(
        "📊 Usar DI Futuro (1Y) como Benchmark",
        value=saved_benchmark_config.get('usar_di_futuro', False),
        key="usar_di_futuro_checkbox",
        help=f"Quando marcado, o benchmark será o DI Futuro 1Y ({taxa_di_1y*100:.2f}%) ao invés do cálculo híbrido. Os targets das carteiras serão: Conservador = DI+1pp, Moderado = DI+2pp, Agressivo = DI+3pp."
    )
    
    st.session_state['usar_di_futuro'] = usar_di_futuro
    
    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        st.markdown("**Ponderação**")
        
        # Seleção do período histórico
        periodo_historico = st.radio(
            "Período Histórico de Referência:",
            options=['5 Anos', 'Período Completo (Desde 2005)'],
            index=0 if saved_benchmark_config['periodo_historico'] == '5 Anos' else 1,
            key="periodo_historico",
            disabled=usar_di_futuro
        )
        
        # Define qual média usar baseado na seleção
        if periodo_historico == '5 Anos':
            media_hist_selecionada = media_5y_ex_post
            label_periodo = "5 Anos"
        else:
            media_hist_selecionada = media_total_ex_post
            label_periodo = "Período Completo"
        
        peso_hist = st.slider("Peso Histórico (%)", 0, 100, int(saved_benchmark_config['peso_hist']), 5, key="peso_hist_slider", disabled=usar_di_futuro) / 100
        peso_mkt = 1 - peso_hist
        if not usar_di_futuro:
            st.caption(f"Histórico ({label_periodo}): {peso_hist*100:.0f}% | Mercado (Ex-Ante): {peso_mkt*100:.0f}%")
        else:
            st.caption("⚠️ Ponderação desabilitada - usando DI Futuro 1Y")
    
    if usar_di_futuro:
        # Usa DI Futuro 1Y nominal (PRE) direto da ETTJ 252 dias úteis
        benchmark_calc = taxa_di_1y
    else:
        benchmark_calc = (media_hist_selecionada * peso_hist) + (taxa_real_ex_ante * peso_mkt)
    
    with col_in2:
        if usar_di_futuro:
            st.markdown("**Benchmark: DI Futuro (1Y)**")
            st.metric(
                "Target Base (DI 1Y Nominal - PRE)", 
                f"{benchmark_calc*100:.2f}%",
                delta="DI Futuro PRE",
                help=f"Benchmark baseado no DI Futuro 1Y ({taxa_di_1y*100:.2f}% nominal) extraído da ETTJ PRE para 252 dias úteis. Os perfis usarão: Conservador = DI+1pp, Moderado = DI+2pp, Agressivo = DI+3pp."
            )
        else:
            st.markdown("**Benchmark Calculado (IPCA + X%)**")
            st.metric(
                "Target Final Sugerido", 
                f"{benchmark_calc*100:.2f}%",
                help=f"Meta de juro real híbrida calculada como: ({peso_hist*100:.0f}% × {media_hist_selecionada*100:.2f}% histórico) + ({peso_mkt*100:.0f}% × {taxa_real_ex_ante*100:.2f}% mercado). Esta será a meta de retorno real das carteiras (IPCA + X%). Carteiras conservadoras terão meta inferior, agressivas superior."
            )
        
        override = st.checkbox(
            "Sobrepor valor calculado manualmente?", 
            value=saved_benchmark_config['override'], 
            key="override_checkbox",
            disabled=usar_di_futuro,
            help="Marque esta opção para ignorar o cálculo automático e definir uma meta customizada. Útil quando você tem uma expectativa específica que difere do modelo."
        )
        if override:
            benchmark_final = st.number_input(
                "Target Manual (%)", 
                value=float(saved_benchmark_config['valor_manual']), 
                key="valor_manual_input",
                help="Meta de juro real customizada em % ao ano. Exemplo: 6.0% significa meta de IPCA + 6% a.a."
            ) / 100
        else:
            benchmark_final = benchmark_calc

    st.session_state['benchmark_final'] = benchmark_final
    
    # Botões salvar/restaurar benchmark
    col_save_bench, col_reset_bench = st.columns([1, 1])
    with col_save_bench:
        if st.button("💾 Salvar Configuração do Benchmark", use_container_width=True, key="save_benchmark"):
            st.session_state['benchmark_config_salvo'] = {
                'periodo_historico': periodo_historico,
                'peso_hist': peso_hist * 100,
                'override': override,
                'valor_manual': benchmark_final * 100 if override else benchmark_calc * 100,
                'usar_di_futuro': usar_di_futuro
            }
            st.success("✅ Configuração do Benchmark salva!")
            st.rerun()
    with col_reset_bench:
        if st.button("🔄 Restaurar Padrões", use_container_width=True, key="reset_benchmark"):
            if 'benchmark_config_salvo' in st.session_state:
                del st.session_state['benchmark_config_salvo']
                st.success("✅ Configuração padrão restaurada!")
                st.rerun()
    
    if not df_hist.empty:
        st.markdown("**Evolução Histórica do Juro Real (Janela Móvel 12 Meses)**")
        st.line_chart(df_hist['Juro_Real_12m'] * 100)

# === PÁGINA 2: CENÁRIOS ===
elif pagina == "Cenários Macro":
    st.title("Premissas e Cenários Macroeconômicos")
    
    bench_ref = st.session_state.get('benchmark_final', 0.06)
    st.info(f"O Benchmark definido na etapa anterior foi **IPCA + {bench_ref*100:.2f}%**. As projeções abaixo devem buscar consistência com este alvo.")

    st.markdown("""
    Esta etapa utiliza uma abordagem **Top-Down**. Primeiro, definimos os cenários para as variáveis macroeconômicas (Drivers).
    Em seguida, o modelo calcula automaticamente o retorno esperado de cada classe de ativo aplicando regras de spread e marcação a mercado.
    """)

    st.subheader("1. Probabilidades dos Cenários")
    st.caption("Distribuição de probabilidade para os cenários Bear, Neutro e Bull nos próximos 12 meses.")
    
    # Carrega valores salvos ou usa defaults
    default_probs = {'bear': 45.0, 'neutro': 10.0, 'bull': 45.0}
    saved_probs = st.session_state.get('probabilidades_salvas', default_probs)
    
    col_prob1, col_prob2, col_prob3, col_check = st.columns(4)
    with col_prob1: 
        prob_bear = st.number_input(
            "Prob. Bear (%)", 
            0.0, 100.0, saved_probs['bear'], 5.0, 
            key="prob_bear",
            help="Probabilidade de ocorrência do cenário pessimista (Bear). Considera recessão, crise política, aperto monetário severo, fuga de capital, etc."
        ) / 100
    with col_prob2: 
        prob_neutro = st.number_input(
            "Prob. Neutro (%)", 
            0.0, 100.0, saved_probs['neutro'], 5.0, 
            key="prob_neutro",
            help="Probabilidade de manutenção do status quo. Crescimento moderado, inflação controlada, sem grandes mudanças macroeconômicas."
        ) / 100
    with col_prob3: 
        prob_bull = st.number_input(
            "Prob. Bull (%)", 
            0.0, 100.0, saved_probs['bull'], 5.0, 
            key="prob_bull",
            help="Probabilidade do cenário otimista (Bull). Reforma estrutural, aceleração do crescimento, queda de juros, melhora fiscal, rally de ativos."
        ) / 100
    
    soma = prob_bear + prob_neutro + prob_bull
    with col_check:
        st.markdown("**Validação**")
        if abs(soma - 1.0) < 0.01: st.success("Soma: 100%")
        else: st.error(f"Soma: {soma*100:.0f}% (Ajuste)")
    
    # Botões salvar/restaurar probabilidades
    col_save_prob, col_reset_prob = st.columns([1, 1])
    with col_save_prob:
        if st.button("💾 Salvar Probabilidades", use_container_width=True, key="save_prob"):
            st.session_state['probabilidades_salvas'] = {
                'bear': prob_bear * 100,
                'neutro': prob_neutro * 100,
                'bull': prob_bull * 100
            }
            st.success("✅ Probabilidades salvas!")
            st.rerun()
    with col_reset_prob:
        if st.button("🔄 Restaurar Padrões", use_container_width=True, key="reset_prob"):
            if 'probabilidades_salvas' in st.session_state:
                del st.session_state['probabilidades_salvas']
                st.success("✅ Probabilidades padrão restauradas!")
                st.rerun()

    st.divider()

    st.subheader("2. Parâmetros de Mercado")
    st.markdown("""
    Para calcular o ganho ou perda de capital (fechamento/abertura de curva) na Renda Fixa Pré e IPCA, 
    é necessário comparar as taxas projetadas nos cenários com as taxas de mercado vigentes hoje, considerando a duration.
    """)
    
    # Info sobre as taxas da ETTJ
    st.info(f"📊 Taxas atualizadas da ETTJ: Pré 504d ({st.session_state['data_pre_504']}) = {st.session_state['taxa_pre_504']*100:.2f}% | "
           f"Real 1638d ({st.session_state['data_real_1638']}) = {st.session_state['taxa_real_1638']*100:.2f}%")
    
    # Debug ETTJ
    with st.expander("🔍 Debug ETTJ (Investigar discrepâncias)", expanded=False):
        # Botão para limpar cache e recarregar taxas ETTJ
        if st.button("🔄 Limpar Cache e Recarregar Taxas ETTJ"):
            get_taxa_pre_vertice_504.clear()
            get_taxa_real_vertice_1638.clear()
            if 'taxa_pre_504' in st.session_state:
                del st.session_state['taxa_pre_504']
            if 'taxa_real_1638' in st.session_state:
                del st.session_state['taxa_real_1638']
            st.rerun()
        
        if 'debug_ettj_pre' in st.session_state:
            st.write("**Taxa Pré (504 dias):**")
            debug_pre = st.session_state['debug_ettj_pre']
            st.write(f"Data: {debug_pre['data']}")
            st.write(f"Colunas disponíveis: {debug_pre['colunas']}")
            st.write(f"Vértice 504 existe exato? {debug_pre.get('vertice_504_existe', 'N/A')}")
            st.write(f"Taxa retornada: {debug_pre.get('taxa_retornada', 'N/A')}")
            st.write(f"Foi interpolado? {debug_pre.get('foi_interpolado', 'N/A')}")
            st.write(f"Range dias: {debug_pre.get('min_dias', '?')} - {debug_pre.get('max_dias', '?')}")
            if 'vertice_504_row' in debug_pre and debug_pre['vertice_504_row']:
                st.write("Vértice 504 encontrado:")
                st.json(debug_pre['vertice_504_row'])
            st.write("Curva completa (sample):")
            if 'sample_completo' in debug_pre:
                st.dataframe(pd.DataFrame(debug_pre['sample_completo']))
        else:
            st.info("Debug vazio. Clique no botão acima para recarregar as taxas ETTJ.")
        
        if 'debug_ettj_real' in st.session_state:
            st.write("**Taxa Real (1638 dias):**")
            debug_real = st.session_state['debug_ettj_real']
            st.write(f"Data: {debug_real['data']}")
            st.write(f"Colunas disponíveis: {debug_real['colunas']}")
            st.write(f"Vértice 1638 existe exato? {debug_real.get('vertice_1638_existe', 'N/A')}")
            st.write(f"Taxa retornada: {debug_real.get('taxa_retornada', 'N/A')}")
            st.write(f"Foi interpolado? {debug_real.get('foi_interpolado', 'N/A')}")
            st.write(f"Range dias: {debug_real.get('min_dias', '?')} - {debug_real.get('max_dias', '?')}")
            if 'vertice_1638_row' in debug_real and debug_real['vertice_1638_row']:
                st.write("Vértice 1638 encontrado:")
                st.json(debug_real['vertice_1638_row'])
            st.write("Curva completa (sample):")
            if 'sample_completo' in debug_real:
                st.dataframe(pd.DataFrame(debug_real['sample_completo']))
        
        if 'debug_ettj_pre_error' in st.session_state:
            st.error(f"Erro Pré: {st.session_state['debug_ettj_pre_error']}")
        if 'debug_ettj_real_error' in st.session_state:
            st.error(f"Erro Real: {st.session_state['debug_ettj_real_error']}")
    
    # Carrega valores salvos ou usa defaults (agora com taxas da ETTJ)
    default_params = {
        'taxa_pre': st.session_state['taxa_pre_504']*100,  # Vértice 504
        'duration_pre': 2.0,
        'taxa_real': st.session_state['taxa_real_1638']*100,  # Vértice 1638
        'duration_imab': 6.53
    }
    saved_params = st.session_state.get('parametros_mercado_salvos', default_params)
    
    with st.expander("Expandir Configurações de Taxas e Duration", expanded=True):
        col_mkt1, col_mkt2, col_mkt3, col_mkt4 = st.columns(4)
        taxa_pre_mercado = col_mkt1.number_input(
            "Taxa Pré (Nominal) Hoje %", 
            value=saved_params['taxa_pre'], 
            format="%.2f", 
            key="taxa_pre", 
            help="Taxa pré-fixada de mercado extraída do vértice 504 dias úteis da ETTJ ANBIMA. Usada como referência para calcular ganho/perda de capital no IRF-M quando a curva se move nos cenários."
        )
        duration_pre = col_mkt2.number_input(
            "Duration Pré (Anos)", 
            value=saved_params['duration_pre'], 
            key="dur_pre",
            help="Duration modificada da carteira de títulos pré-fixados (IRF-M). Representa a sensibilidade do preço a variações de 1% na taxa. Exemplo: Duration 2 anos → queda de 1% na taxa = ganho de 2% no preço."
        )
        taxa_real_mercado = col_mkt3.number_input(
            "Taxa Real (NTN-B) Hoje %", 
            value=saved_params['taxa_real'], 
            format="%.2f", 
            key="taxa_real", 
            help="Taxa real (IPCA+) de mercado extraída do vértice 1638 dias úteis da ETTJ ANBIMA (DI x IPCA 252). Usada para calcular marcação a mercado do IMA-B quando a taxa real muda nos cenários."
        )
        duration_imab = col_mkt4.number_input(
            "Duration IMA-B (Anos)", 
            value=saved_params['duration_imab'], 
            key="dur_imab",
            help="Duration modificada da carteira de títulos indexados ao IPCA (IMA-B). Sensibilidade do preço a variações de 1% na taxa real. Exemplo: Duration 6.5 anos → queda de 1% na taxa real = ganho de 6.5% no preço."
        )
    
    # Botões salvar/restaurar parâmetros
    col_save_params, col_reset_params = st.columns([1, 1])
    with col_save_params:
        if st.button("💾 Salvar Parâmetros", use_container_width=True, key="save_params"):
            st.session_state['parametros_mercado_salvos'] = {
                'taxa_pre': taxa_pre_mercado,
                'duration_pre': duration_pre,
                'taxa_real': taxa_real_mercado,
                'duration_imab': duration_imab
            }
            st.success("✅ Parâmetros salvos!")
            st.rerun()
    with col_reset_params:
        if st.button("🔄 Restaurar Padrões", use_container_width=True, key="reset_params"):
            if 'parametros_mercado_salvos' in st.session_state:
                del st.session_state['parametros_mercado_salvos']
                st.success("✅ Parâmetros padrão restaurados!")
                st.rerun()

    st.subheader("3. Premissas Macroeconômicas")
    st.caption("Preencha as expectativas para as variáveis-chave em cada cenário. Estas premissas alimentarão o pricing dos ativos.")
    
    # Carrega valores salvos ou usa defaults
    drivers = ["CDI", "IPCA", "Juro Real (NTN-B)", "Juro Nominal (2 anos)", "Ibovespa (Nominal)", "IFIX (Nominal)"]
    default_macro = pd.DataFrame(index=drivers)
    default_macro['Bear'] = [14.50, 5.50, 8.50, 15.0, -20.0, -10.0]
    default_macro['Neutro'] = [13.00, 4.00, 7.35, 13.0, 20.0, 10.0]
    default_macro['Bull'] = [11.00, 3.80, 5.50, 10.0, 40.0, 25.0]
    
    if 'drivers_macro_salvos' in st.session_state:
        df_macro = st.session_state['drivers_macro_salvos'].copy()
    else:
        df_macro = default_macro.copy()

    df_macro_edit = st.data_editor(df_macro, use_container_width=True, key="drivers_macro")
    
    # Botões salvar/restaurar drivers
    col_save_drivers, col_reset_drivers = st.columns([1, 1])
    with col_save_drivers:
        if st.button("💾 Salvar premissas", use_container_width=True, key="save_drivers"):
            st.session_state['drivers_macro_salvos'] = df_macro_edit.copy()
            st.success("✅ Premissas macroeconômicas salvas!")
            st.rerun()
    with col_reset_drivers:
        if st.button("🔄 Restaurar Padrões", use_container_width=True, key="reset_drivers"):
            if 'drivers_macro_salvos' in st.session_state:
                del st.session_state['drivers_macro_salvos']
                st.success("✅ Premissas padrão restauradas!")
                st.rerun()

    def calcular_retornos(cenario_dict, nome_cenario):
        cdi = cenario_dict['CDI'] / 100
        ipca = cenario_dict['IPCA'] / 100
        juro_real_cenario = cenario_dict['Juro Real (NTN-B)'] / 100
        juro_nominal_cenario = cenario_dict['Juro Nominal (2 anos)'] / 100
        ibov = cenario_dict['Ibovespa (Nominal)'] / 100
        ifix = cenario_dict['IFIX (Nominal)'] / 100
        
        diff_taxa_real = (taxa_real_mercado/100) - juro_real_cenario
        ganho_capital_real = duration_imab * diff_taxa_real
        r_imab = ((1 + ganho_capital_real) * (1 + juro_real_cenario) * (1 + ipca)) - 1
        
        diff_taxa_pre = (taxa_pre_mercado/100) - juro_nominal_cenario
        ganho_capital_pre = duration_pre * diff_taxa_pre
        r_irfm = ((1 + ganho_capital_pre) * (1 + juro_nominal_cenario)) - 1
        
        return {
            "CDI (Caixa)": cdi,
            "IDA-DI (Crédito)": cdi * (1 + 0.01),
            "IMA-B (Inflação)": r_imab,
            "IRF-M (Pré-fixado)": r_irfm,
            "IHFA (Multimercado)": cdi * 1.15,
            "IBOV (Ações BR)": ibov,
            "IFIX (FIIs)": ifix,
            "IMA-B 5+ (Ilíquidos)": ((1 + 0.12) * (1 + ipca)) - 1,
        }

    resultados = {}
    for cen in ['Bear', 'Neutro', 'Bull']:
        resultados[cen] = calcular_retornos(df_macro_edit[cen].to_dict(), cen)

    df_ativos = pd.DataFrame(resultados)
    df_ativos['Retorno Esperado (%)'] = ((df_ativos['Bear']*prob_bear) + (df_ativos['Neutro']*prob_neutro) + (df_ativos['Bull']*prob_bull)) * 100
    
    for col in ['Bear', 'Neutro', 'Bull']:
        df_ativos[col] = df_ativos[col] * 100
        
    st.divider()
    
    st.subheader("4. Matriz de Retornos Projetada")
    st.markdown("""
    A tabela abaixo apresenta a rentabilidade nominal esperada para cada classe de ativo, derivada dos cenários macro acima.
    O **Retorno Esperado (Ponderado)** será utilizado como input para a otimização de Markowitz.
    """)
    
    st.dataframe(
        df_ativos,
        column_config={
            "Retorno Esperado (%)": st.column_config.ProgressColumn(
                "Retorno Esperado (Ponderado)",
                format="%.2f%%",
                min_value=-5,
                max_value=25,
                help="Média ponderada pelas probabilidades dos cenários",
            ),
            "Bear": st.column_config.NumberColumn(format="%.2f%%"),
            "Neutro": st.column_config.NumberColumn(format="%.2f%%"),
            "Bull": st.column_config.NumberColumn(format="%.2f%%"),
        },
        use_container_width=True
    )
    
    st.session_state['premissas_retorno'] = df_ativos

# === PÁGINA 3: OTIMIZAÇÃO ===
elif pagina == "Otimização":
    st.title("Otimização de Portfólio")
    
    if 'premissas_retorno' not in st.session_state:
        st.error("⚠️ Atenção: Você precisa definir os Cenários na Fase B antes de prosseguir.")
        st.stop()
        
    df_premissas = st.session_state['premissas_retorno']
    bench_base = st.session_state.get('benchmark_final', 0.06)
    
    # Seletor de MÉTODO principal
    st.markdown("### 🎯 Método de Otimização")
    metodo_principal = st.radio(
        "Escolha o método:",
        options=["Markowitz (MVO)", "Black-Litterman", "Risk Parity"],
        help="""
        **Markowitz**: Otimização clássica baseada em retorno/risco histórico
        **Black-Litterman**: Combina equilíbrio de mercado com suas views subjetivas
        **Risk Parity**: Equaliza contribuição de risco (ignora retornos esperados)
        """,
        horizontal=True
    )
    
    # Sub-opções para Markowitz
    if metodo_principal == "Markowitz (MVO)":
        modo_markowitz = st.radio(
            "Critério de otimização:",
            options=["Máximo Sharpe Ratio", "Mínima Variância"],
            help="**Máximo Sharpe Ratio**: Melhor relação retorno/risco | **Mínima Variância**: Menor volatilidade",
            horizontal=True,
            key="modo_markowitz"
        )
    
    # === EXPLICAÇÃO DOS MÉTODOS ===
    with st.expander("📖 Entenda os Métodos de Otimização", expanded=False):
        st.markdown("""
        ### **1. Markowitz (Mean-Variance Optimization)**
        - **Input**: Retornos esperados + Matriz de covariância
        - **Output**: Carteira que maximiza Sharpe ou minimiza variância
        - **Prós**: Simples, intuitivo, base da teoria moderna de portfólio
        - **Contras**: Sensível a erros nas estimativas de retorno (garbage in, garbage out)
        
        ---
        
        ### **2. Black-Litterman**
        - **Input**: Equilíbrio de mercado (market caps) + Views subjetivas do gestor + Covariância
        - **Output**: Retornos ajustados que combinam mercado e opinião, aplicados no Markowitz
        - **Prós**: Incorpora views qualitativas, reduz extremos, mais estável
        - **Contras**: Requer definição de market caps e confiança nas views
        - **Quando usar**: Quando você tem convicções fortes sobre alguns ativos (ex: "IBOV vai render 20%")
        
        ---
        
        ### **3. Risk Parity**
        - **Input**: Apenas matriz de covariância (ignora retornos esperados!)
        - **Output**: Carteira onde cada ativo contribui igualmente para o risco total
        - **Prós**: Não depende de estimativas de retorno, muito estável, bom para diversificação
        - **Contras**: Pode alocar muito em ativos de baixo retorno/baixa volatilidade
        - **Quando usar**: Quando não confia nas projeções de retorno ou busca máxima diversificação
        
        ---
        
        **💡 Dica**: Rode os 3 métodos e compare os resultados! Muitos gestores usam uma combinação dos três.
        """)
    
    st.markdown("""
    Esta etapa calcula a alocação ótima utilizando as premissas de retorno definidas na Fase B
    e uma matriz de covariância histórica para estimar o risco.
    """)
    
    st.divider()
    
    # === BLACK-LITTERMAN: Interface para Views ===
    if metodo_principal == "Black-Litterman":
        st.markdown("### 📝 Definição de Views (Black-Litterman)")
        st.markdown("""
        Defina suas expectativas (views) sobre o desempenho de ativos específicos.
        O modelo combinará suas views com o equilíbrio de mercado para gerar retornos ajustados.
        """)
        
        ativos = df_premissas.index.tolist()
        
        # Pesos de equilíbrio (market caps) - pode ser ajustado manualmente
        st.subheader("Pesos de Equilíbrio (Market Cap)")
        st.caption("Distribua 100% entre os ativos conforme a representatividade de mercado.")
        
        default_market_caps = [0.0, 30.0, 25.0, 10.0, 10.0, 15.0, 5.0, 5.0]  # Exemplo
        
        if 'market_caps_bl' not in st.session_state:
            st.session_state['market_caps_bl'] = default_market_caps
        
        df_market_caps = pd.DataFrame({
            "Ativo": ativos,
            "Market Cap (%)": st.session_state['market_caps_bl']
        })
        
        df_market_caps_edit = st.data_editor(
            df_market_caps,
            column_config={
                "Market Cap (%)": st.column_config.NumberColumn(min_value=0, max_value=100, format="%.1f%%")
            },
            use_container_width=True,
            hide_index=True,
            key="market_caps_editor"
        )
        
        soma_market = df_market_caps_edit['Market Cap (%)'].sum()
        if abs(soma_market - 100) > 0.1:
            st.error(f"⚠️ Soma dos Market Caps: {soma_market:.1f}% (deve ser 100%)")
        else:
            st.success(f"✓ Soma: {soma_market:.1f}%")
            st.session_state['market_caps_bl'] = df_market_caps_edit['Market Cap (%)'].tolist()
        
        st.divider()
        
        # Interface para adicionar views
        st.subheader("Views Subjetivas")
        st.caption("Adicione suas expectativas de retorno para ativos específicos (ex: 'IBOV vai render 20% no próximo ano')")
        
        # Inicializa lista de views
        if 'bl_views' not in st.session_state:
            st.session_state['bl_views'] = []
        
        # Formulário para adicionar nova view
        with st.expander("➕ Adicionar Nova View", expanded=len(st.session_state['bl_views']) == 0):
            col_ativo, col_ret, col_conf = st.columns([2, 1, 1])
            view_ativo = col_ativo.selectbox("Ativo", ativos, key="new_view_ativo")
            view_retorno = col_ret.number_input("Retorno Esperado (%)", value=15.0, format="%.2f", key="new_view_ret")
            view_confianca = col_conf.slider("Confiança", 1, 10, 5, help="1=Baixa, 10=Alta", key="new_view_conf")
            
            if st.button("Adicionar View", key="add_view_btn"):
                st.session_state['bl_views'].append({
                    'ativo': view_ativo,
                    'retorno': view_retorno,
                    'confianca': view_confianca
                })
                st.rerun()
        
        # Mostra views adicionadas
        if st.session_state['bl_views']:
            st.markdown("**Views Configuradas:**")
            for i, view in enumerate(st.session_state['bl_views']):
                col1, col2 = st.columns([4, 1])
                col1.info(f"**{view['ativo']}**: Retorno esperado de **{view['retorno']:.2f}%** (Confiança: {view['confianca']}/10)")
                if col2.button("🗑️", key=f"del_view_{i}"):
                    st.session_state['bl_views'].pop(i)
                    st.rerun()
        else:
            st.warning("⚠️ Nenhuma view configurada. O modelo usará apenas o equilíbrio de mercado.")
    
    # === RISK PARITY: Configurações ===
    elif metodo_principal == "Risk Parity":
        st.markdown("### ⚖️ Configurações Risk Parity")
        st.info("""
        Risk Parity equaliza a **contribuição de risco** de cada ativo, não seus pesos.
        Ativos menos voláteis terão pesos maiores para equilibrar o risco total.
        """)
        
        use_vol_target = st.checkbox("Definir Target de Volatilidade", value=False, 
                                     help="Se marcado, ajusta alavancagem para atingir volatilidade específica")
        
        if use_vol_target:
            vol_target_rp = st.number_input("Volatilidade Alvo (% a.a.)", value=8.0, min_value=1.0, max_value=30.0, format="%.1f") / 100
        else:
            vol_target_rp = None

    # 1. Definição de Perfil
    st.divider()
    st.markdown("### 1. Definição de Perfil e Restrições")
    st.caption("Selecione o perfil de risco. O Target de Retorno e as restrições de alocação serão ajustados automaticamente.")
    
    col_perf, col_target = st.columns([1, 2])
    
    with col_perf:
        perfil_selecionado = st.radio("Selecione o Perfil:", ["Conservador", "Moderado", "Agressivo"])
    
    # Verifica se está usando DI Futuro como benchmark
    usar_di_futuro_opt = st.session_state.get('usar_di_futuro', False)
    
    if usar_di_futuro_opt:
        # Quando usa DI Futuro: Conservador = DI+1pp, Moderado = DI+2pp, Agressivo = DI+3pp
        if perfil_selecionado == "Conservador":
            target_final = bench_base + 0.01
            defaults_min = [0.0, 0.50, 0.10, 0.0, 0.05, 0.0, 0.0, 0.0]
            defaults_max = [0.0, 0.80, 0.40, 0.20, 0.20, 0.10, 0.0, 0.0]
            delta_str = "DI + 1.00 p.p."
            vol_min, vol_max = 0.01, 0.02  # 1% a 2%
        elif perfil_selecionado == "Moderado":
            target_final = bench_base + 0.02
            defaults_min = [0.0, 0.20, 0.10, 0.0, 0.05, 0.05, 0.05, 0.0]
            defaults_max = [0.0, 0.60, 0.40, 0.20, 0.25, 0.20, 0.10, 0.10]
            delta_str = "DI + 2.00 p.p."
            vol_min, vol_max = 0.03, 0.04  # 3% a 4%
        else: # Agressivo
            target_final = bench_base + 0.03
            defaults_min = [0.0, 0.10, 0.10, 0.0, 0.05, 0.10, 0.05, 0.05]
            defaults_max = [0.0, 0.40, 0.40, 0.20, 0.30, 0.35, 0.15, 0.15]
            delta_str = "DI + 3.00 p.p."
            vol_min, vol_max = 0.05, 0.06  # 5% a 6%
    else:
        # Quando usa benchmark híbrido (padrão)
        if perfil_selecionado == "Conservador":
            target_final = bench_base - 0.01
            defaults_min = [0.0, 0.50, 0.10, 0.0, 0.05, 0.0, 0.0, 0.0]
            defaults_max = [0.0, 0.80, 0.40, 0.20, 0.20, 0.10, 0.0, 0.0]
            delta_str = "- 1.00 p.p."
            vol_min, vol_max = 0.01, 0.02  # 1% a 2%
        elif perfil_selecionado == "Moderado":
            target_final = bench_base
            defaults_min = [0.0, 0.20, 0.10, 0.0, 0.05, 0.05, 0.05, 0.0]
            defaults_max = [0.0, 0.60, 0.40, 0.20, 0.25, 0.20, 0.10, 0.10]
            delta_str = "Benchmark"
            vol_min, vol_max = 0.03, 0.04  # 3% a 4%
        else: # Agressivo
            target_final = bench_base + 0.01
            defaults_min = [0.0, 0.10, 0.10, 0.0, 0.05, 0.10, 0.05, 0.05]
            defaults_max = [0.0, 0.40, 0.40, 0.20, 0.30, 0.35, 0.15, 0.15]
            delta_str = "+ 1.00 p.p."
            vol_min, vol_max = 0.05, 0.06  # 5% a 6%
    
    # Armazena limites de volatilidade no session_state
    st.session_state['vol_min'] = vol_min
    st.session_state['vol_max'] = vol_max

    ipca_focus = expectativa_ipca
    
    # Quando usa DI Futuro, o target já está nominal, não precisa somar IPCA
    if usar_di_futuro_opt:
        target_nominal = target_final
    else:
        target_nominal = ((1 + target_final) * (1 + ipca_focus)) - 1

    with col_target:
        if usar_di_futuro_opt:
            st.metric(
                label=f"Meta de Retorno Nominal ({perfil_selecionado})", 
                value=f"{target_final*100:.2f}%",
                delta=delta_str
            )
            st.caption(f"Meta Nominal (para o Solver): {target_nominal*100:.2f}%")
        else:
            st.metric(
                label=f"Meta de Retorno Real ({perfil_selecionado})", 
                value=f"{target_final*100:.2f}% + IPCA",
                delta=delta_str
            )
            st.caption(f"Meta Nominal Equivalente (para o Solver): {target_nominal*100:.2f}%")
        
        st.info(f"📊 **Banda de Volatilidade:** {vol_min*100:.0f}% a {vol_max*100:.0f}% a.a.")

    # Tabela de Constraints (com persistência por perfil)
    st.subheader(f"Limites de Alocação: {perfil_selecionado}")
    ativos = df_premissas.index.tolist()
    
    # Inicializa storage de restrições se não existir
    if 'constraints_por_perfil' not in st.session_state:
        st.session_state['constraints_por_perfil'] = {}
    
    # Carrega restrições salvas ou usa defaults
    if perfil_selecionado in st.session_state['constraints_por_perfil']:
        df_constraints = st.session_state['constraints_por_perfil'][perfil_selecionado].copy()
    else:
        df_constraints = pd.DataFrame({
            "Ativo": ativos,
            "Min (%)": [x * 100 for x in defaults_min],
            "Max (%)": [x * 100 for x in defaults_max]
        })
    
    df_constraints_edit = st.data_editor(
        df_constraints,
        column_config={
            "Min (%)": st.column_config.NumberColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Max (%)": st.column_config.NumberColumn(min_value=0, max_value=100, format="%.1f%%")
        },
        use_container_width=True, hide_index=True,
        key=f"constraints_{perfil_selecionado}"
    )
    
    # Botões para gerenciar restrições
    col_save, col_reset, col_unrestricted = st.columns([1, 1, 1])
    with col_save:
        if st.button("💾 Salvar Restrições", use_container_width=True):
            st.session_state['constraints_por_perfil'][perfil_selecionado] = df_constraints_edit.copy()
            st.success(f"✅ Restrições salvas para o perfil {perfil_selecionado}!")
            st.rerun()
    
    with col_reset:
        if st.button("🔄 Restaurar Padrões", use_container_width=True):
            if perfil_selecionado in st.session_state['constraints_por_perfil']:
                del st.session_state['constraints_por_perfil'][perfil_selecionado]
                st.success(f"✅ Restrições padrão restauradas para {perfil_selecionado}!")
                st.rerun()
    
    with col_unrestricted:
        if st.button("🔓 Sem Restrições", use_container_width=True, help="Define Min=0% e Max=100% para todos os ativos"):
            df_unrestricted = df_constraints_edit.copy()
            df_unrestricted['Min (%)'] = 0.0
            df_unrestricted['Max (%)'] = 100.0
            st.session_state['constraints_por_perfil'][perfil_selecionado] = df_unrestricted
            st.success("✅ Restrições removidas (Min=0%, Max=100%)!")
            st.rerun()

    st.divider()

    # 2. Dados e Cálculo
    st.markdown("### 2. Otimização de Carteira")
    if metodo_principal == "Markowitz (MVO)":
        if modo_markowitz == "Máximo Sharpe Ratio":
            st.markdown("O sistema buscará a combinação de ativos que **maximiza o Sharpe Ratio** (melhor relação retorno/risco) respeitando os limites definidos.")
        else:
            st.markdown("O sistema buscará a combinação de ativos com **menor volatilidade possível**, respeitando os limites definidos.")
    elif metodo_principal == "Black-Litterman":
        st.markdown("O modelo combinará suas views com o equilíbrio de mercado para gerar alocação otimizada.")
    else:
        st.markdown("O modelo equalizará a **contribuição de risco** de cada ativo na carteira.")
    
    # Botão para baixar dados da API (uma vez só)
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("📥 Baixar Dados da API", use_container_width=True):
            with st.spinner("Conectando Comdinheiro (Recuperando cotações)..."):
                user = st.session_state.get('api_user')
                pwd = st.session_state.get('api_pass')
                df_returns_temp, msg = fetch_comdinheiro_data(user, pwd)
                
                # Mostrar dados brutos da API
                if 'api_full_response' in st.session_state:
                    with st.expander("🔍 RESPOSTA COMPLETA DA API (Debug)", expanded=False):
                        full_resp = st.session_state['api_full_response']
                        st.error(f"⚠️ TAB-P0: {full_resp['total_rows_p0']} linhas | TAB-P1: {full_resp['total_rows_p1']} linhas")
                        st.write(f"**Chaves do JSON:** {full_resp['raw_json_keys']}")
                        st.write(f"**Chaves em 'resposta':** {full_resp['resposta_keys']}")
                
                if df_returns_temp is None:
                    st.error(f"Falha na API: {msg}")
                else:
                    # Armazena dados no session_state para reutilização
                    st.session_state['df_returns_api'] = df_returns_temp
                    st.session_state['api_download_time'] = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.success(f"✅ Dados recuperados com sucesso! {msg}")
                    st.rerun()
    
    # Mostra status dos dados
    if 'df_returns_api' in st.session_state:
        st.info(f"✓ Dados carregados às {st.session_state.get('api_download_time', 'N/A')} - {len(st.session_state['df_returns_api'])} dias")
    else:
        st.warning("⚠️ Dados não carregados. Clique em 'Baixar Dados da API' primeiro.")
        st.stop()
    
    # Botão para calcular otimização (usa dados já baixados)
    with col_btn2:
        calcular_btn = st.button("🎯 Calcular Otimização", type="secondary", use_container_width=True)
    
    if calcular_btn:
        with st.spinner("Calculando otimização..."):
            # Usa os dados já carregados do session_state
            df_returns = st.session_state['df_returns_api']
            
            # ALINHAMENTO CRÍTICO: Garantir mesma ordem entre premissas e dados históricos
            ativos_esperados = df_premissas.index.tolist()
            ativos_api = df_returns.columns.tolist()
            
            # Verifica se todos os ativos esperados estão na API
            ativos_faltantes = [a for a in ativos_esperados if a not in ativos_api]
            if ativos_faltantes:
                st.error(f"Ativos não encontrados na API: {ativos_faltantes}")
                st.caption(f"Ativos disponíveis: {ativos_api}")
                st.stop()
            
            # Reordena df_returns para corresponder à ordem de df_premissas
            df_returns_aligned = df_returns[ativos_esperados]
            
            # Matriz Covariância (Dados já são retornos diários, anualiza multiplicando por 252)
            cov_matrix = df_returns_aligned.cov() * 252
            
            # Preparação Solver - agora alinhados
            mu = df_premissas['Retorno Esperado (%)'].values / 100
            S = cov_matrix.values
            num_assets = len(mu)
            
            # Debug info (opcional - mostra para usuário)
            with st.expander("📊 Informações de Debug", expanded=False):
                st.write(f"**Ativos (ordem):** {ativos_esperados}")
                st.write(f"**Retornos Esperados (mu):** {[f'{m*100:.2f}%' for m in mu]}")
                st.write(f"**Target Nominal:** {target_nominal*100:.2f}%")
                st.write(f"**Shape cov_matrix:** {S.shape}")
                st.write(f"**Dados históricos:** {len(df_returns_aligned)} dias")
                
                st.write("**Amostra dos dados históricos (últimos 5 dias):**")
                st.dataframe(df_returns_aligned.tail())
                
                st.write("**Estatísticas dos retornos históricos diários:**")
                st.dataframe(df_returns_aligned.describe())
                
                st.write("**Diagonal da matriz de covariância (variâncias anualizadas):**")
                variancias = np.diag(S)
                vols = np.sqrt(variancias)
                st.write({ativos_esperados[i]: f"{vols[i]*100:.2f}%" for i in range(len(ativos_esperados))})
            
            min_bounds = df_constraints_edit['Min (%)'].values / 100
            max_bounds = df_constraints_edit['Max (%)'].values / 100
            bounds = tuple(zip(min_bounds, max_bounds))
            
            def portfolio_vol(weights, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def neg_sharpe_ratio(weights, mu, cov_matrix, risk_free=0.0):
                """Sharpe Ratio negativo para minimização (retorno - rf) / volatilidade"""
                ret = np.dot(weights, mu)
                vol = portfolio_vol(weights, cov_matrix)
                return -(ret - risk_free) / vol if vol > 0 else 1e10
            
            # Chute inicial: pesos iguais
            init_guess = num_assets * [1. / num_assets,]
            
            # Usa CDI como proxy de risk-free (média histórica anualizada)
            risk_free_rate = df_returns_aligned.iloc[:, 0].mean() * 252 if 'CDI' in ativos_esperados[0] else 0.0
            
            # 2.1. OTIMIZAÇÃO: Executa método selecionado
            try:
                # ========== MÉTODO 1: MARKOWITZ ==========
                if metodo_principal == "Markowitz (MVO)":
                    if modo_markowitz == "Máximo Sharpe Ratio":
                        # Maximizar Sharpe Ratio com restrições de volatilidade
                        vol_min_perfil = st.session_state.get('vol_min', 0.01)
                        vol_max_perfil = st.session_state.get('vol_max', 0.10)
                        
                        constraints = [
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'ineq', 'fun': lambda x: portfolio_vol(x, S) - vol_min_perfil},  # vol >= vol_min
                            {'type': 'ineq', 'fun': lambda x: vol_max_perfil - portfolio_vol(x, S)}   # vol <= vol_max
                        ]
                        
                        opt_result = minimize(
                            neg_sharpe_ratio, 
                            init_guess, 
                            args=(mu, S, risk_free_rate), 
                            method='SLSQP', 
                            bounds=bounds, 
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9}
                        )
                    else:
                        # Minimizar Variância com restrições de volatilidade
                        vol_min_perfil = st.session_state.get('vol_min', 0.01)
                        vol_max_perfil = st.session_state.get('vol_max', 0.10)
                        
                        constraints = [
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'ineq', 'fun': lambda x: portfolio_vol(x, S) - vol_min_perfil},  # vol >= vol_min
                            {'type': 'ineq', 'fun': lambda x: vol_max_perfil - portfolio_vol(x, S)}   # vol <= vol_max
                        ]
                        
                        opt_result = minimize(
                            portfolio_vol,
                            init_guess,
                            args=(S,),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9}
                        )
                    
                    metodo_usado = f"Markowitz ({modo_markowitz})"
                
                # ========== MÉTODO 2: BLACK-LITTERMAN ==========
                elif metodo_principal == "Black-Litterman":
                    # Preparar inputs do Black-Litterman
                    market_caps = np.array(st.session_state['market_caps_bl']) / 100
                    
                    # Construir matriz de views (P) e vetor de retornos esperados (Q)
                    if st.session_state['bl_views']:
                        num_views = len(st.session_state['bl_views'])
                        P = np.zeros((num_views, num_assets))
                        Q = np.zeros(num_views)
                        omega_diag = np.zeros(num_views)
                        
                        for i, view in enumerate(st.session_state['bl_views']):
                            ativo_idx = ativos_esperados.index(view['ativo'])
                            P[i, ativo_idx] = 1.0  # View absoluta sobre um ativo
                            Q[i] = view['retorno'] / 100
                            # Omega: incerteza da view (inverso da confiança)
                            omega_diag[i] = 0.01 / view['confianca']  # Quanto maior confiança, menor incerteza
                        
                        omega = np.diag(omega_diag)
                    else:
                        # Sem views: usa apenas equilíbrio de mercado (Pi)
                        P = np.zeros((1, num_assets))
                        P[0, 0] = 1.0  # View dummy
                        Q = np.array([0.0])
                        omega = np.diag([1e10])  # Incerteza infinita = ignora a view
                    
                    # Parâmetro tau (incerteza no equilíbrio)
                    tau = 0.025
                    
                    # Executa Black-Litterman
                    mu_bl = black_litterman(S, market_caps, tau, P, Q, omega)
                    
                    # Agora usa os retornos BL no Markowitz (Máximo Sharpe) com restrições de volatilidade
                    vol_min_perfil = st.session_state.get('vol_min', 0.01)
                    vol_max_perfil = st.session_state.get('vol_max', 0.10)
                    
                    constraints = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'ineq', 'fun': lambda x: portfolio_vol(x, S) - vol_min_perfil},  # vol >= vol_min
                        {'type': 'ineq', 'fun': lambda x: vol_max_perfil - portfolio_vol(x, S)}   # vol <= vol_max
                    ]
                    
                    opt_result = minimize(
                        neg_sharpe_ratio, 
                        init_guess, 
                        args=(mu_bl, S, risk_free_rate), 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    # Atualiza mu para usar nos cálculos de retorno
                    mu = mu_bl
                    metodo_usado = "Black-Litterman"
                
                # ========== MÉTODO 3: RISK PARITY ==========
                elif metodo_principal == "Risk Parity":
                    # Risk Parity ignora restrições de Min/Max e retornos esperados
                    weights_rp = risk_parity_optimization(S, vol_target_rp)
                    
                    # Cria objeto fake de resultado para compatibilidade
                    class FakeResult:
                        def __init__(self, x):
                            self.x = x
                            self.success = True
                            self.fun = 0
                    
                    opt_result = FakeResult(weights_rp)
                    metodo_usado = "Risk Parity"
                
                if opt_result.success:
                    weights_opt = opt_result.x
                    ret_otimo = np.dot(weights_opt, mu)
                    vol_otima = portfolio_vol(weights_opt, S)
                    sharpe_otimo = (ret_otimo - risk_free_rate) / vol_otima if vol_otima > 0 else 0
                    
                    # 2.2. GERAÇÃO DA FRONTEIRA
                    ret_min_possible = np.min(mu)
                    ret_max_possible = np.max(mu)
                    target_range = np.linspace(ret_min_possible, ret_max_possible, 20)
                    
                    frontier_vols = []
                    frontier_rets = []
                    
                    for t_ret in target_range:
                        cons_loop = (
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
                            {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - t_ret}
                        )
                        res = minimize(portfolio_vol, init_guess, args=(S,), method='SLSQP', bounds=bounds, constraints=cons_loop)
                        if res.success:
                            frontier_vols.append(res.fun)
                            frontier_rets.append(t_ret)
                    
                    # --- PLOTAGEM ---
                    col_chart, col_data = st.columns([2, 1])
                    
                    with col_chart:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Fronteira eficiente (apenas para Markowitz e BL)
                        if metodo_principal != "Risk Parity":
                            ax.plot([v*100 for v in frontier_vols], [r*100 for r in frontier_rets], 'b--', label='Fronteira Eficiente')
                        
                        ax.scatter(vol_otima*100, ret_otimo*100, color='red', s=150, zorder=5, label=f'Carteira {perfil_selecionado}')
                        
                        # Marca o benchmark do perfil
                        ax.axhline(y=target_nominal*100, color='green', linestyle=':', linewidth=2, label='Benchmark Perfil')
                        
                        ax.set_title(f"Risco x Retorno ({perfil_selecionado}) - {metodo_usado}")
                        ax.set_xlabel("Volatilidade Esperada (% a.a.)")
                        ax.set_ylabel("Retorno Esperado (% a.a.)")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                    with col_data:
                        st.info(f"**Método:** {metodo_usado}")
                        st.success(f"**Volatilidade:** {vol_otima*100:.2f}%")
                        st.metric("Retorno Otimizado", f"{ret_otimo*100:.2f}%", 
                                 delta=f"{(ret_otimo - target_nominal)*100:+.2f} p.p. vs target")
                        st.metric("Sharpe Ratio", f"{sharpe_otimo:.2f}")
                        
                        # Info adicional para Black-Litterman
                        if metodo_principal == "Black-Litterman":
                            st.caption(f"✓ {len(st.session_state['bl_views'])} view(s) aplicada(s)")
                        
                        # Info adicional para Risk Parity
                        if metodo_principal == "Risk Parity":
                            st.caption("✓ Contribuição de risco equalizada")
                        
                        st.markdown("**Alocação Sugerida:**")
                        df_w = pd.DataFrame({"Ativo": ativos, "Peso": weights_opt*100})
                        
                        # Se Risk Parity, adiciona coluna de contribuição de risco
                        if metodo_principal == "Risk Parity":
                            portfolio_var = weights_opt.T @ S @ weights_opt
                            portfolio_vol = np.sqrt(portfolio_var)
                            marginal_contrib = S @ weights_opt
                            risk_contrib = weights_opt * marginal_contrib / portfolio_vol if portfolio_vol > 0 else weights_opt * 0
                            risk_contrib_pct = (risk_contrib / np.sum(risk_contrib)) * 100 if np.sum(risk_contrib) > 0 else risk_contrib * 0
                            
                            df_w['Contribuição Risco (%)'] = risk_contrib_pct
                            st.dataframe(
                                df_w.style.format({"Peso": "{:.2f}%", "Contribuição Risco (%)": "{:.2f}%"})
                                    .background_gradient(subset=['Peso'], cmap="Blues")
                                    .background_gradient(subset=['Contribuição Risco (%)'], cmap="Greens"),
                                use_container_width=True, hide_index=True
                            )
                        else:
                            st.dataframe(
                                df_w.style.format({"Peso": "{:.2f}%"}).background_gradient(cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )
                    
                    # === MATRIZ DE CORRELAÇÃO (no final da página) ===
                    st.divider()
                    st.subheader("📊 Matriz de Correlação Histórica dos Ativos")
                    st.markdown("""
                    Matriz de correlação calculada com base nos retornos históricos diários.
                    Valores próximos de **+1** indicam movimentos similares, **-1** indicam movimentos opostos, e **0** indica baixa correlação.
                    """)
                    
                    df_returns_corr = st.session_state['df_returns_api']
                    
                    # Calcula matriz de correlação
                    corr_matrix = df_returns_corr.corr()
                    
                    # Exibe com formatação de heatmap
                    st.dataframe(
                        corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1, axis=None)
                                         .format("{:.2f}"),
                        use_container_width=True
                    )
                    
                    # Estatísticas adicionais
                    with st.expander("📈 Análise de Correlações"):
                        st.write("**Pares com maior correlação positiva:**")
                        # Pega triângulo superior sem diagonal
                        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        corr_pairs = upper_tri.stack().sort_values(ascending=False)
                        top_pos = corr_pairs.head(5)
                        for (idx1, idx2), val in top_pos.items():
                            st.write(f"• {idx1} ↔ {idx2}: **{val:.3f}**")
                        
                        st.write("\n**Pares com maior correlação negativa (diversificação):**")
                        top_neg = corr_pairs.tail(5)
                        for (idx1, idx2), val in top_neg.items():
                            st.write(f"• {idx1} ↔ {idx2}: **{val:.3f}**")
                    
                    # --- BOTÃO PARA GERAR RELATÓRIO ---
                    st.markdown("---")
                    if st.button("📄 Gerar Relatório Completo", use_container_width=True, key="gerar_relatorio_btn"):
                        with st.spinner("Gerando relatório..."):
                            # Gera gráfico como imagem base64
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            img_base64 = base64.b64encode(buf.read()).decode()
                            
                            # Monta HTML do relatório
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Relatório Asset Allocation - {perfil_selecionado}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                                .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                                h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
                                h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }}
                                h3 {{ color: #34495e; margin-top: 20px; }}
                                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                                th {{ background-color: #1f77b4; color: white; padding: 12px; text-align: left; }}
                                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                                tr:hover {{ background-color: #f5f5f5; }}
                                .metric {{ display: inline-block; background: #e3f2fd; padding: 15px 20px; margin: 10px 10px 10px 0; 
                                         border-radius: 5px; border-left: 4px solid #1f77b4; }}
                                .metric-label {{ font-size: 12px; color: #666; }}
                                .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
                                .chart {{ text-align: center; margin: 20px 0; }}
                                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; 
                                          text-align: center; color: #666; font-size: 12px; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h1>📊 Relatório de Asset Allocation - {perfil_selecionado}</h1>
                                <p><strong>Data de Geração:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                                <p><strong>Método de Otimização:</strong> {metodo_usado}</p>
                                
                                <h2>🎯 Resultados da Otimização</h2>
                                <div>
                                    <div class="metric">
                                        <div class="metric-label">Retorno Esperado</div>
                                        <div class="metric-value">{ret_otimo*100:.2f}%</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-label">Volatilidade</div>
                                        <div class="metric-value">{vol_otima*100:.2f}%</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-label">Sharpe Ratio</div>
                                        <div class="metric-value">{sharpe_otimo:.2f}</div>
                                    </div>
                                </div>
                                
                                <h3>Alocação Recomendada</h3>
                                <table>
                                    <tr><th>Ativo</th><th>Peso Alocado</th></tr>
                                    {''.join([f'<tr><td>{ativo}</td><td>{peso:.2f}%</td></tr>' for ativo, peso in zip(ativos, weights_opt*100)])}
                                </table>
                                
                                <h2>📈 Fronteira Eficiente</h2>
                                <div class="chart">
                                    <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;">
                                </div>
                                
                                <h2>🎲 Premissas de Cenários</h2>
                                <h3>Probabilidades dos Cenários</h3>
                                <table>
                                    <tr><th>Perfil</th><th>Bear (%)</th><th>Neutro (%)</th><th>Bull (%)</th></tr>
                                    {pd.DataFrame(st.session_state.get('probabilidades_salvas', {{}})).to_html(index=False, escape=False, header=False, border=0)}
                                </table>
                                
                                <h3>Retornos Esperados por Cenário</h3>
                                {df_premissas.to_html(index=False, border=1)}
                                
                                <h2>⚙️ Parâmetros de Mercado</h2>
                                <table>
                                    <tr><th>Parâmetro</th><th>Valor</th></tr>
                                    <tr><td>Taxa Pré (Nominal)</td><td>{st.session_state.get('parametros_mercado_salvos', {{}}).get('taxa_pre', 12):.2f}%</td></tr>
                                    <tr><td>Duration Pré</td><td>{st.session_state.get('parametros_mercado_salvos', {{}}).get('duration_pre', 2):.2f} anos</td></tr>
                                    <tr><td>Taxa Real (NTN-B)</td><td>{st.session_state.get('parametros_mercado_salvos', {{}}).get('taxa_real', 6):.2f}%</td></tr>
                                    <tr><td>Duration IMA-B</td><td>{st.session_state.get('parametros_mercado_salvos', {{}}).get('duration_imab', 6):.2f} anos</td></tr>
                                </table>
                                
                                <h2>🔒 Restrições de Alocação</h2>
                                {df_constraints_edit[['Ativo', 'Min (%)', 'Max (%)']].to_html(index=False, border=1)}
                                
                                <h2>📊 Estatísticas Históricas</h2>
                                <h3>Volatilidades Anualizadas</h3>
                                <table>
                                    <tr><th>Ativo</th><th>Volatilidade</th></tr>
                                    {''.join([f'<tr><td>{ativos_esperados[i]}</td><td>{vols[i]*100:.2f}%</td></tr>' for i in range(len(ativos_esperados))])}
                                </table>
                                
                                <div class="footer">
                                    <p>Relatório gerado automaticamente pelo sistema Ghia MFO - Asset Allocation</p>
                                    <p>© {datetime.now().year} - Todos os direitos reservados</p>
                                </div>
                            </div>
                        </body>
                        </html>
                        """
                            
                            # Salva o arquivo localmente
                            filename = f"{timestamp}_{perfil_selecionado}_relatorio.html"
                            
                            try:
                                with open(filename, 'w', encoding='utf-8') as f:
                                    f.write(html_content)
                                st.success(f"✅ Relatório salvo localmente: **{filename}**")
                            except Exception as e:
                                st.warning(f"Não foi possível salvar localmente: {e}")
                            
                            # Salva em session_state para persistir após rerun
                            st.session_state['ultimo_relatorio'] = {
                                'html': html_content,
                                'filename': filename,
                                'timestamp': timestamp
                            }
                    
                    # Botão de download (sempre visível se houver relatório gerado)
                    if 'ultimo_relatorio' in st.session_state:
                        rel = st.session_state['ultimo_relatorio']
                        st.download_button(
                            label="⬇️ Download do Último Relatório Gerado",
                            data=rel['html'],
                            file_name=rel['filename'],
                            mime="text/html",
                            key="download_relatorio_btn",
                            use_container_width=True
                        )
                        
                else:
                    st.error("Solução Inviável: Não foi possível encontrar uma alocação ótima.")
                    st.caption("Dica: Relaxe as restrições de Min/Max.")
                    
                    # Debug detalhado
                    with st.expander("🔍 Diagnóstico Detalhado", expanded=True):
                        st.write(f"**Mensagem do Solver:** {opt_result.message}")
                        st.write(f"**Status:** {opt_result.status}")
                        st.write(f"**Retornos esperados dos ativos:**")
                        for i, ativo in enumerate(ativos_esperados):
                            st.write(f"  - {ativo}: {mu[i]*100:.2f}%")
                        st.write(f"**Soma dos limites mínimos:** {np.sum(min_bounds)*100:.1f}%")
            
            except Exception as e:
                st.error(f"Erro no Solver: {e}")
                st.exception(e)

## Para atualizar o codigo online, digitar no console:
# cd "C:\Users\GabrielHenriqueMarti\Desktop\Asset Allocation"

# 1. Adiciona as alterações
# git add .

# 2. Cria um commit com descrição
# git commit -m "Descrição da alteração feita"

# 3. Envia para o GitHub
# git push
