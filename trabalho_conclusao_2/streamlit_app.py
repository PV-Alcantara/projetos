# streamlit_app.py

import streamlit as st
import os
import shutil
import glob
import re
import requests
import pandas as pd
import numpy as np
from lxml import etree
from pandasql import sqldf
from dotenv import load_dotenv
import io




from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURAÇÃO DE PASTAS E VARIÁVEIS GLOBAIS ---
docs_folder = "trabalho_conclusao_2/docs_tcc/"
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

# Inicializa o dicionário de DataFrames na session_state
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {}

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO ---

def parse_xml(xml_path):
    # Lógica idêntica à do script original
    try:
        if isinstance(xml_path, str):
            tree = etree.parse(xml_path)
        else: # Trata o arquivo de upload do Streamlit
            tree = etree.parse(xml_path)

        root = tree.getroot()
        ns = {'nfe': 'http://www.portalfiscal.inf.br/nfe'}
        extracted_rows = []

        uf = root.xpath('//nfe:emit//nfe:enderEmit/nfe:UF/text()', namespaces=ns)
        cidade = root.xpath('//nfe:emit//nfe:enderEmit/nfe:xMun/text()', namespaces=ns)
        cnpj = root.xpath('//nfe:emit//nfe:CNPJ/text()', namespaces=ns)
        nome_fantasia = root.xpath('//nfe:emit//nfe:xFant/text()', namespaces=ns)
        ie = root.xpath('//nfe:emit//nfe:IE/text()', namespaces=ns)
        crt = root.xpath('//nfe:emit//nfe:CRT/text()', namespaces=ns)
        numero_nf = root.xpath('//nfe:ide/nfe:nNF/text()', namespaces=ns)
        data_emissao_full = root.xpath('//nfe:ide/nfe:dhEmi/text()', namespaces=ns)
        data_emissao = [d.split('T')[0] for d in data_emissao_full] # Formato AAAA-MM-DD
        uf_destino = root.xpath('//nfe:dest/nfe:enderDest/nfe:UF/text()', namespaces=ns)

        for det in root.xpath('//nfe:det', namespaces=ns):
            ncm = det.xpath('.//nfe:prod/nfe:NCM/text()', namespaces=ns)
            cfop = det.xpath('.//nfe:prod/nfe:CFOP/text()', namespaces=ns)
            cest = det.xpath('.//nfe:prod/nfe:CEST/text()', namespaces=ns)
            vProd = det.xpath('.//nfe:prod/nfe:vProd/text()', namespaces=ns)
            descricao = det.xpath('.//nfe:prod/nfe:xProd/text()', namespaces=ns)

            icms_elem = det.xpath('.//nfe:ICMS/*', namespaces=ns)
            vICMS = icms_elem[0].xpath('.//nfe:vICMS/text()', namespaces=ns) if icms_elem else []

            pis_elem = det.xpath('.//nfe:PIS/*', namespaces=ns)
            vPIS = pis_elem[0].xpath('.//nfe:vPIS/text()', namespaces=ns) if pis_elem else []

            cofins_elem = det.xpath('.//nfe:COFINS/*', namespaces=ns)
            vCOFINS = cofins_elem[0].xpath('.//nfe:vCOFINS/text()', namespaces=ns) if cofins_elem else []

            ipi_elem = det.xpath('.//nfe:IPI/*', namespaces=ns)
            vIPI = ipi_elem[0].xpath('.//nfe:vIPI/text()', namespaces=ns) if ipi_elem else []

            extracted_rows.append({
                'numero_nf': numero_nf[0] if numero_nf else None,
                'data_emissao': data_emissao[0] if data_emissao else None,
                'uf': uf[0] if uf else None,
                'uf_destino': uf_destino[0] if uf_destino else None,
                'cidade': cidade[0] if cidade else None,
                'cnpj': cnpj[0] if cnpj else None,
                'nome_fantasia': nome_fantasia[0] if nome_fantasia else None,
                'ie': ie[0] if ie else None,
                'crt': crt[0] if crt else None,
                'descricao': descricao[0] if descricao else None,
                'ncm': ncm[0] if ncm else None,
                'cfop': cfop[0] if cfop else None,
                'cest': cest[0] if cest else None,
                'vProd': vProd[0] if vProd else None,
                'icms': vICMS[0] if vICMS else '0',
                'pis': vPIS[0] if vPIS else '0',
                'cofins': vCOFINS[0] if vCOFINS else '0',
                'ipi': vIPI[0] if vIPI else '0'
            })

        df_xml = pd.DataFrame(extracted_rows)
        df_xml['chave_uf'] = df_xml['uf'] + df_xml['uf_destino']
        return df_xml

    except Exception as e:
        # print(f"Erro ao processar XML: {e}") # Evita print no Streamlit
        return pd.DataFrame()

def limpar_nome_coluna(nome):
    nome = pd.Series(nome).str.normalize('NFKD').str.cat()
    nome = re.sub(r'[^a-zA-Z0-9\s]', '', nome)
    nome = nome.lower().strip().replace(' ', '_')
    return nome

@st.cache_data
def carregar_e_processar_bases_estaticas(docs_folder):
    """Carrega, limpa e processa os DataFrames auxiliares."""
    dfs_bases = {}
    status_success = True

    # --- NCM ---
    tabela_ncm = os.path.join(docs_folder, "Tabela_NCM.csv")
    try:
        df_ncm = pd.read_csv(tabela_ncm, skiprows=4)
        df_ncm['NCM'] = df_ncm['Código'].str.replace('.', '', regex=False)
        df_ncm.columns = [limpar_nome_coluna(col) for col in df_ncm.columns]
        dfs_bases['df_ncm'] = df_ncm
    except Exception:
        st.error("Erro ao carregar 'Tabela_NCM.csv'. Verifique o arquivo.")
        status_success = False

    # --- ICMS ---
    tabela_icms = os.path.join(docs_folder, "uf_icms.csv")
    try:
        df = pd.read_csv(tabela_icms, sep=',')
        id_vars_name = df.columns[0]
        value_vars = df.columns[1:]
        df_melted = df.melt(
            id_vars=id_vars_name,
            value_vars=value_vars,
            var_name='UF (destino)',
            value_name='Valor'
        )
        df_melted = df_melted.rename(columns={id_vars_name: 'UF (origem)'})
        df_icms = df_melted[['UF (origem)', 'UF (destino)', 'Valor']]
        df_icms['chave_uf'] = df_icms['UF (origem)'] + df_icms['UF (destino)']
        dfs_bases['df_icms'] = df_icms
    except Exception:
        st.error("Erro ao carregar 'uf_icms.csv'. Verifique o arquivo.")
        status_success = False

    # --- IPI ---
    tabela_ipi = os.path.join(docs_folder, "Tipi.csv")
    try:
        df_ipi = pd.read_csv(tabela_ipi, skiprows=7, sep=';')
        df_ipi.columns = [limpar_nome_coluna(col) for col in df_ipi.columns]
        df_ipi['codigo'] = df_ipi['ncm'].copy()
        df_ipi['ncm'] = df_ipi['ncm'].str.replace('.', '', regex=False)
        df_ipi['aliquota'] = df_ipi['aliquota'].astype(str).str.replace('NT','0',regex=False).fillna('0')
        #df_ipi = df_ipi.drop(columns=[col for col in ['unnamed_4','ex'] if col in df_ipi.columns], errors='ignore')
        dfs_bases['df_ipi'] = df_ipi
    except Exception:
        st.error("Erro ao carregar 'Tipi.csv'. Verifique o arquivo.")
        status_success = False

    # --- CNAE ---
    tabela_cnae = os.path.join(docs_folder, "Tabela_CNAE.csv")
    try:
        df_cnae = pd.read_csv(tabela_cnae, encoding='latin-1', sep=';')
        df_cnae.columns = [limpar_nome_coluna(col) for col in df_cnae.columns]
        dfs_bases['df_cnae'] = df_cnae
    except Exception:
        st.error("Erro ao carregar 'Tabela_CNAE.csv'. Verifique o arquivo.")
        status_success = False

    if not status_success:
        return {}

    return dfs_bases

# --- FERRAMENTAS DO AGENTE (Modificadas para usar st.session_state['dfs']) ---

@tool
def sql_documentos(query_sql: str) -> pd.DataFrame:
    """
    Esse deve ser o último recurso a ser utilizado.
    *NUNCA* use essa ferramenta para faze cálculos, apenas utilize-a para fazer buscas nos dataframes disponíveis.
    """
    try:
        # Acessa o dicionário 'dfs' do session_state
        dfs_session = st.session_state.get('dfs', {})
        return sqldf(query_sql, dfs_session)
    except Exception as e:
        return pd.DataFrame({'ERRO': [f"Erro na query SQL: {e}"]})

@tool
def consulta_cnae_cnpj(cnpj: str):
    """Ferramenta destinada a utilizar a trazer o código CNAE a partir de um CNPJ.
    Sempre utilize essa ferramenta para trazer o código cnae de um cnpj desejado. **NUNCA** inclua outros caractéres como letras, espaços, pontos, barras ou aspas.

    entrada esperada:
        -> cnpj = valor numérico (string)

    exemplo correto: 34234567000101
    exemplo incorreto: 34.234.567/0001-01
    
    """
    cnpj_digits = "".join(filter(str.isdigit, str(cnpj)))
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj_digits}"
    resultado_falha = {"cnpj": cnpj_digits, "cnae_fiscal": "NAO_ENCONTRADO", "status": "FALHA_NA_CONSULTA"}

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return resultado_falha
        data = resp.json()
        if "cnae_fiscal" in data:
            return {"cnpj": cnpj_digits, "cnae_fiscal": data["cnae_fiscal"], "status": "SUCESSO"}
        return resultado_falha
    except:
        return resultado_falha

@tool
def mapear_coluna_por_chave(chave_coluna: str, novos_dados: list, nome_nova_coluna: str) -> str:
    """
    Ferramenta para mapear valores de uma lista de resultados (CNPJ, CNAE, etc.) 
    para a tabela 'df_notas', usando uma coluna chave de correspondência.

    Esta ferramenta simula um 'JOIN' para replicar um valor (ex: CNAE) 
    para todas as linhas do df_notas que compartilham a mesma chave (ex: CNPJ).

    Entrada esperada:
      -> chave_coluna (str): Nome da coluna em 'df_notas' a ser usada como chave de busca 
                             (ex: 'cnpj', 'numero_nf').
      -> novos_dados (list): Lista de dicionários com a chave e o novo valor.
                            Ex: [{'cnpj': '123...', 'cnae': '4700'}, {'cnpj': '456...', 'cnae': '5600'}]
      -> nome_nova_coluna (str): O nome da nova coluna a ser criada no 'df_notas'
                                 (também deve ser uma chave presente nos dicionários de 'novos_dados').
    """
    if 'dfs' not in st.session_state or 'df_notas' not in st.session_state['dfs']:
        return "ERRO: df_notas não disponível"

    df_notas = st.session_state['dfs']['df_notas']
    try:
        df_map = pd.DataFrame(novos_dados)
    except:
        return "ERRO: 'novos_dados' não está no formato de lista de dicionários"

    if chave_coluna not in df_map.columns:
        return f"ERRO: coluna '{chave_coluna}' não encontrada no df_map"
    
    # Lógica de renomeação idêntica à do script original (corrigida a verificação)
    if nome_nova_coluna not in df_map.columns:
        valores_presentes = [col for col in df_map.columns if col != chave_coluna]
        if len(valores_presentes) == 1:
            df_map = df_map.rename(columns={valores_presentes[0]: nome_nova_coluna})
        else:
            return f"ERRO: coluna '{nome_nova_coluna}' não encontrada nos dados"

    df_map = df_map[[chave_coluna, nome_nova_coluna]].drop_duplicates(subset=[chave_coluna])
    
    if nome_nova_coluna in df_notas.columns:
        df_notas = df_notas.drop(columns=[nome_nova_coluna])
        
    df_notas = df_notas.merge(df_map, on=chave_coluna, how='left')
    st.session_state['dfs']['df_notas'] = df_notas
    return f"SUCESSO: coluna '{nome_nova_coluna}' mapeada e adicionada ao 'df_notas'. O df_notas possui agora {len(df_notas)} linhas."


@tool
def calcular_icms(valor_mercadoria: float, chave_uf: str) -> float:
    """Ferramenta destinada a calcular o valor do ICMS para UMA ÚNICA mercadoria.
    Sempre use esta ferramenta para obter o valor do ICMS a ser pago.

    Entrada esperada:
      -> valor_mercadoria = valor numérico (int ou float) do produto/serviço.
      -> chave_uf = string de 4 caracteres (UF_Origem + UF_Destino).

    Exemplo correto dos inputs: 105.50, SPSC
    """
    if 'dfs' not in st.session_state or 'df_icms' not in st.session_state['dfs']:
        # Levanta um erro que o agente pode capturar
        raise ValueError("DataFrame df_icms não disponível na sessão.") 

    df_icms = st.session_state['dfs']['df_icms']
    linha = df_icms[df_icms['chave_uf'] == chave_uf.upper()]
    
    if linha.empty:
        # Lança erro para ser tratado pelo Agente (ideal)
        raise ValueError(f"Alíquota não encontrada para a chave_uf: {chave_uf}")
        
    aliquota = float(linha['Valor'].iloc[0]) / 100
    return float(valor_mercadoria) * aliquota

@tool
def calcular_pis_cofins(valor_mercadoria: float, valor_icms: float) -> tuple[float, float]:
    """Ferramenta destinada a realizar o cálculo dos impostos PIS e COFINS.
    Usa as alíquotas PIS (1.65%) e COFINS (7.6%) sobre a Base de Cálculo (Valor Mercadoria - ICMS).
    Retorna a tupla (valor_pis, valor_cofins).
    """
    ALIQUOTA_PIS = 0.0165
    ALIQUOTA_COFINS = 0.076
    base = float(valor_mercadoria) - float(valor_icms)
    # Garante que a base não seja negativa
    if base < 0:
        base = 0.0
    
    valor_pis = base * ALIQUOTA_PIS
    valor_cofins = base * ALIQUOTA_COFINS
    
    return valor_pis, valor_cofins


@tool
def desc_cnae(cnae: str) -> str:
    """Ferramenta destinada a trazer o campo 'nome_setor' que é a descrição do ramo de atuação de um código CNAE
    sempre utilize essa ferramenta para pesquisar descrições de CNAE somente com os números, **NUNCA** inclua outros caractéres como pontos, virgulas,letras.

    entrada esperada -> código CNAE

    exemplo correto: '4763601'
    exemplo incorreto: '476.36.01'
        
    """
    if 'dfs' not in st.session_state or 'df_cnae' not in st.session_state['dfs']:
        return "CNAE Não Encontrado"
    df_cnae = st.session_state['dfs']['df_cnae']
    linha = df_cnae[df_cnae['cnae'] == str(cnae)]
    return linha['nome_setor'].iloc[0] if not linha.empty else "CNAE Não Encontrado"

@tool
def salvar_resultados_no_dataframe(valores_calculados: list, nome_coluna: str) -> str:
    """
    Ferramenta GENÉRICA para salvar uma lista de valores calculados (ICMS, PIS, COFINS, IPI, etc.) 
    em uma nova coluna do DataFrame 'df_notas'.

    Esta ferramenta é o passo final para persistir *qualquer* resultado de cálculo na base de dados.

    Entrada esperada:
      -> valores_calculados (list): A lista de todos os valores calculados (deve ter o mesmo 
            número de elementos que o df_notas).
      -> nome_coluna (str): O nome exato da nova coluna a ser criada (ex: 'pis_calculado', 'icms_teorico').
            **NÃO** inclua espaços, acentos ou caracteres especiais no nome da coluna.
            
    Exemplo de uso: salvar_resultados_no_dataframe(valores_calculados=[1.0, 2.5, 3.0], nome_coluna='pis_calculado')
    """
    if 'dfs' not in st.session_state or 'df_notas' not in st.session_state['dfs']:
        return "ERRO: df_notas não disponível"
        
    df_notas = st.session_state['dfs']['df_notas']
    
    # Validação do tamanho
    if len(df_notas) != len(valores_calculados):
        return (f"ERRO: O número de resultados ({len(valores_calculados)}) "
                f"não corresponde ao número de linhas do df_notas ({len(df_notas)}). "
                "Revise o cálculo.")
                
    # Adiciona a nova coluna
    try:
        df_notas[nome_coluna] = valores_calculados
        st.session_state['dfs']['df_notas'] = df_notas # Atualiza o estado
    except Exception as e:
        return f"ERRO ao salvar no DataFrame: {e}"
        
    return f"SUCESSO: A coluna '{nome_coluna}' foi adicionada ao DataFrame 'df_notas' com {len(valores_calculados)} valores."

@tool
def consultar_aliquota_ipi(ncm: str) -> float:
    """
    Busca a alíquota de IPI (Imposto sobre Produtos Industrializados) para um único código NCM.

    A alíquota é retornada como um percentual (ex: 10.0 para 10%).

    Entrada esperada:
      -> ncm = String contendo apenas os dígitos do NCM (ex: '85044021').

    Exemplo correto dos inputs: '85044021'
    """
    if 'dfs' not in st.session_state or 'df_ipi' not in st.session_state['dfs']:
        return 0.0
    df_ipi = st.session_state['dfs']['df_ipi']
    
    ncm_limpo = str(ncm).strip()
    linha = df_ipi[df_ipi['ncm'] == ncm_limpo]
    
    if linha.empty:
        return 0.0
        
    aliquota_str = linha['aliquota'].iloc[0]
    
    try:
        # Retorna o valor em porcentagem (ex: 10.0) e não em decimal
        aliquota_ipi = float(aliquota_str) 
        return aliquota_ipi 
    except:
        return 0.0

# --- CONFIGURAÇÃO DO AGENTE LANGCHAIN ---

# Removido st.cache_resource da função setup_agent para garantir que use a memory correta no Streamlit
def setup_agent(chave_api):
    # Obtém a memória da session_state
    if 'agent_memory' not in st.session_state:
         st.session_state['agent_memory'] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.25,
        api_key=chave_api,
        max_tokens=800
    )

    tools = [
        desc_cnae,
        calcular_icms,
        calcular_pis_cofins,
        consulta_cnae_cnpj,
        sql_documentos,
        salvar_resultados_no_dataframe,
        consultar_aliquota_ipi,
        mapear_coluna_por_chave
    ]

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """você é um especialista em contabilidade financeiro nas regras fiscasis do Brasil. sua função é fazer o cálculo de impostos das notas fiscais eletronicas fornecidas e validar se o que foi executado faz sentido com o documento.
        Sua principal prioridade é a segurança e a precisão da informação.
             
        Suas regras:
        ->Responder de forma cordial ao consumidor.
        ->Sempre utilizar as ferramentas adequadas para executar os cálculos fiscais.
        ->**NUNCA** tentar realizar cálculos que não estejam estipulados nas ferramentas.
        ->Se a busca não retornar informações relevantes, informe de forma amigável que a informação não foi encontrada em sua base de conhecimento.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=st.session_state['agent_memory'],
    verbose=True,
    handle_parsing_errors=True
    )

    return agent_executor

# --- FLUXO DE EXECUÇÃO COM STREAMLIT ---

# Função para realizar os ajustes finais, idêntica à do script original
def aplicar_ajustes_finais(df_notas: pd.DataFrame) -> pd.DataFrame:
    """Aplica a conversão de tipos e a lógica de ajuste de valores 0 após o cálculo do agente."""
    
    colunas_originais = ['icms', 'pis', 'cofins', 'ipi', 'vProd']

    # 1. Ajuste de tipos de dados (o agente pode retornar strings)
    for col in colunas_originais:
        if col in df_notas.columns:
            # Converte para string e substitui vírgulas (,) por pontos (.)
            df_notas[col] = df_notas[col].astype(str).str.replace(',', '.', regex=False)
            # Converte para float. 'coerce' transforma textos inválidos em NaN
            df_notas[col] = pd.to_numeric(df_notas[col], errors='coerce').fillna(0.0) # Preenche NaN com 0 para segurança

    # 2. Lógica de ajuste de valores 0 (Se o original é 0, o calculado deve ser 0)
    colunas_calculadas = {
        'icms': 'icms_calculado',
        'pis': 'pis_calculado',
        'cofins': 'cofins_calculado',
        'ipi': 'ipi_calculado'
    }

    for col_orig, col_calc in colunas_calculadas.items():
        if col_orig in df_notas.columns and col_calc in df_notas.columns:
            # Se o campo original é 0, força o campo calculado a ser 0.0
            df_notas.loc[df_notas[col_orig] == 0.0, col_calc] = 0.0
            
    return df_notas

def executar_fluxo_agente(_agent_executor, input_1, input_2, input_3, input_4):
    """Executa os 4 passos sequenciais e aplica os ajustes finais."""
    st.info("Iniciando o Fluxo de Cálculos Fiscais...")

    try:
        # --- EXECUÇÃO DOS 4 PASSOS ---
        with st.spinner("Passo 1/4: Calculando ICMS..."):
            _agent_executor.invoke({"input": input_1, "chat_history": st.session_state['agent_memory'].buffer_as_messages})
            st.session_state['agent_memory'].clear() # Limpa a memória após o passo 1
            st.success("ICMS calculado e persistido.")
        
        with st.spinner("Passo 2/4: Calculando PIS e COFINS..."):
            _agent_executor.invoke({"input": input_2, "chat_history": st.session_state['agent_memory'].buffer_as_messages})
            st.session_state['agent_memory'].clear() # Limpa a memória após o passo 2
            st.success("PIS e COFINS calculados e persistidos.")
        
        with st.spinner("Passo 3/4: Consultando Alíquotas IPI..."):
            _agent_executor.invoke({"input": input_3, "chat_history": st.session_state['agent_memory'].buffer_as_messages})
            st.session_state['agent_memory'].clear() # Limpa a memória após o passo 3
            st.success("Alíquotas IPI consultadas e persistidas.")
        
        with st.spinner("Passo 4/4: Mapeando CNAE e segmento..."):
            _agent_executor.invoke({"input": input_4, "chat_history": st.session_state['agent_memory'].buffer_as_messages})
            st.session_state['agent_memory'].clear() # Limpa a memória após o passo 4
            st.success("CNAE e segmento mapeados com sucesso.")
            
        # --- AJUSTES FINAIS DO DATAFRAME (IDÊNTICOS AO SEU SCRIPT ORIGINAL) ---
        df_notas_temp = st.session_state['dfs']['df_notas'].copy()
        df_notas_final = aplicar_ajustes_finais(df_notas_temp)
        st.session_state['dfs']['df_notas'] = df_notas_final # Persiste a versão final

        st.success("✅ Fluxo de análise e ajustes finais concluído com sucesso!")
    
    except Exception as e:
        st.error(f"Erro na execução do agente: {e}")
        
    return st.session_state['dfs']

# --- STREAMLIT APP ---

@st.cache_data
def carregar_notas_por_upload(files):
    """Recebe lista de arquivos XML do usuário via upload e cria o df_notas."""
    lista_de_dfs = []
    for file in files:
        df_individual = parse_xml(file)
        if not df_individual.empty:
            df_individual['nome_arquivo'] = file.name
            lista_de_dfs.append(df_individual)
            
    if lista_de_dfs:
        df_notas = pd.concat(lista_de_dfs, ignore_index=True)
        cnpj_list = df_notas['cnpj'].drop_duplicates()
    else:
        df_notas = pd.DataFrame()
        cnpj_list = pd.Series()
        
    return df_notas, cnpj_list

# --- INTERFACE ---
st.title("Processamento de Notas Fiscais Eletrônics - NFe")

# Carrega variáveis de ambiente e agent
load_dotenv()
api_key = os.getenv("api_key")
if not api_key:
    st.error("Chave api_key não encontrada no .env. Verifique o arquivo.")
    st.stop()
    
# Carrega automaticamente bases estáticas e configura o agente (sem cache_resource para o agente)
st.session_state['dfs'].update(carregar_e_processar_bases_estaticas(docs_folder))
agent_executor = setup_agent(api_key)


uploaded_files = st.file_uploader(
    "📥 Faça upload das notas fiscais XML", 
    type=["xml"], 
    accept_multiple_files=True
)

if uploaded_files:
    # O upload substitui o df_notas e a cnpj_list
    df_notas, cnpj_list = carregar_notas_por_upload(uploaded_files)
    if not df_notas.empty:
        st.session_state['dfs']['df_notas'] = df_notas
        st.session_state['dfs']['cnpj_list'] = cnpj_list # Adiciona a cnpj_list ao DFS
        st.success(f"✅ {len(df_notas)} registros processados de {len(uploaded_files)} arquivos XML.")
    else:
        st.warning("Nenhuma nota fiscal XML válida encontrada nos arquivos enviados.")

# --- INTERAÇÃO COM O AGENTE ---
if 'df_notas' in st.session_state['dfs'] and not st.session_state['dfs']['df_notas'].empty:
    df_notas = st.session_state['dfs']['df_notas']

    st.write("### 🧾 Pré-visualização das notas carregadas:")
    st.dataframe(df_notas.head())

    # Botão para executar o agente
    if st.button("🚀 Executar agente de análise e cálculo"):
        # Mensagens de input idênticas ao seu script original
        input_1 = """Eu preciso que você execute os seguintes passos, em ordem:
1. Obtenha as colunas 'vProd' e 'chave_uf' da tabela 'df_notas'.
2. Para CADA linha da tabela, chame a ferramenta 'calcular_icms' para obter o valor.
3. Após obter o ICMS de todas as linhas, use a ferramenta 'salvar_resultados_no_dataframe' 
   para persistir todos esses valores na tabela 'df_notas' em uma nova coluna chamada 'icms_calculado'.
"""
        input_2 = """Eu preciso que você execute os seguintes passos, em ordem:
1. Obtenha as colunas 'vProd' e 'icms' da tabela 'df_notas'.
2. Para CADA linha da tabela, chame a ferramenta 'calcular_pis_cofins' para obter o valor.
3. Após obter os resultados de PIS e COFINS de todas as linhas, use a ferramenta 'salvar_resultados_no_dataframe' 
   para persistir todos esses valores na tabela 'df_notas' em duas novas colunas chamadas 'pis_calculado' e 'cofins_calculado'.
"""
        input_3 = """Eu preciso que você execute os seguintes passos, em ordem:
1. Obtenha a coluna 'ncm' da tabela 'df_notas'.
2. Para CADA linha da tabela, chame a ferramenta **'consultar_aliquota_ipi'** para obter o valor da alíquota de IPI correspondente.
3. Após obter o IPI de todas as linhas, use a ferramenta 'salvar_resultados_no_dataframe'
   para persistir todos esses valores (que são as alíquotas) na tabela 'df_notas' em uma nova coluna chamada **'ipi_calculado'**.
"""
        input_4 = """Eu preciso que você execute os seguintes passos, em ordem:
1. **Primeiro Passo: Obter o CNAE para todos os CNPJs distintos.**
   - Obtenha a lista de CNPJs únicos da 'cnpj_list'. Para isso, use a ferramenta 'sql_documentos' com a query 'SELECT DISTINCT cnpj FROM df_notas'.
   - Para CADA CNPJ dessa lista, chame a ferramenta 'consulta_cnae_cnpj'.
   - Crie uma lista de dicionários de resultados, onde cada dicionário deve conter a chave 'cnpj' e a chave 'cnae_fiscal' (com o valor numérico retornado pela ferramenta).

2. **Segundo Passo: Mapear o CNAE para a tabela principal.**
   - Use a ferramenta **'mapear_coluna_por_chave'** para replicar os valores de CNAE para a tabela 'df_notas'.
   - Use 'cnpj' como a coluna chave (chave_coluna).
   - Use a lista de dicionários do Passo 1 como 'novos_dados'.
   - Chame a nova coluna de 'cnae_fiscal' (nome_nova_coluna).

3. **Terceiro Passo: Obter a descrição do CNAE (segmento).**
   - Use a ferramenta 'sql_documentos' com a query 'SELECT cnae_fiscal FROM df_notas' para obter a lista completa dos códigos CNAE mapeados.
   - Para CADA valor dessa lista, chame a ferramenta 'desc_cnae'.
   - Após obter a descrição (ramo de atuação) de todas as linhas, use a ferramenta 'salvar_resultados_no_dataframe' para persistir todos esses valores na tabela 'df_notas' em uma nova coluna chamada **'segmento'**.
"""
        
        # Execução do fluxo
        dfs_atualizados = executar_fluxo_agente(agent_executor, input_1, input_2, input_3, input_4)
        df_notas_final = dfs_atualizados['df_notas']
        
        st.session_state['df_notas_processado'] = df_notas_final 

        # Exibe o resultado final
        st.write("### 📊 Resultado final (df_notas atualizado):")
        st.dataframe(df_notas_final)


   # Botão para salvar o arquivo final
if 'df_notas_processado' in st.session_state and st.button("💾 Baixar Relatório (relatorio_notas_fiscais.xlsx)"):
    try:
        # **AJUSTE AQUI:** Pegue o DataFrame do session_state
        df_notas_salvar = st.session_state['df_notas_processado'] 
        
        # 1. Cria um buffer de memória (BytesIO)
        buffer = io.BytesIO()
        
        # 2. Salva o DataFrame no buffer de memória
        df_notas_salvar.to_excel(buffer, index=False, engine='openpyxl') 
        
        # 3. Permite o download diretamente do buffer
        st.download_button(
            label="Clique para Baixar o Arquivo Excel",
            data=buffer.getvalue(), # Obtém os dados binários do buffer
            file_name="relatorio_notas_fiscais.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("✅ Buffer de download gerado com sucesso.")

    except Exception as e:
        st.error(f"Erro ao gerar o arquivo Excel: {e}")