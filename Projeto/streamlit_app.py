# streamlit_app.py (VERSÃO FINAL COM PROCESSAMENTO EM LOTES)

import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import datetime
import io
import time
import concurrent.futures

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Ordens de Serviço", layout="wide")

# --- Funções da IA ---

def carregar_modelo_ia():
    """Lê a chave da API dos segredos do Streamlit (se online) ou de um ficheiro local."""
    try:
        # Tenta obter a chave dos segredos do Streamlit (quando está online)
        if 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.info("Chave de API carregada a partir dos segredos online.")
        # Se não estiver online, tenta ler do ficheiro local
        else:
            with open('api_key.txt', 'r') as f:
                api_key = f.read().strip()
            st.info("Chave de API carregada do ficheiro local 'api_key.txt'.")

        if not api_key:
            st.error("Erro: A chave de API não foi encontrada.")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
        
    except FileNotFoundError:
        st.error("Erro Fatal: Ficheiro 'api_key.txt' não encontrado. Crie o ficheiro ou configure os segredos online.")
        return None
    except Exception as e:
        st.error(f"Erro Fatal ao configurar a API do Gemini: {e}")
        return None

def build_gemini_prompt(description):
    """Constrói um prompt refinado para a IA com regras de prioridade e contagem de listas."""

    prompt_final = f"""
    Você é um assistente de IA especialista em extrair dados de Nota de Serviço(NDS). Sua tarefa é analisar o "Texto da NDS", identificar a avaria de MAIOR prioridade e retornar um JSON com a contagem APENAS desse item.

    ---
    ### 1. HIERARQUIA DE PRIORIDADE (REGRA FUNDAMENTAL)
    A análise segue uma ordem estrita. Assim que encontrar uma avaria, pare e conte apenas ela.
    - **Prioridade 1: Postes (PS)**
    - **Prioridade 2: Trafos (TR)**
    - **Prioridade 3: Cabos / Condutores/ Jumpers**
    - **Prioridade 4: Outros**

    ---
    ### 2. REGRAS DE CONTAGEM
    
    * **Para Postes e Trafos:**
      - A contagem principal vem de padrões como `SUBSTITUIR=X` (o valor é X).
      - Se o padrão acima não existir, procure por frases como "TRAZER X POSTE", "LEVAR DOIS TRAFOS", etc. (o valor é o número X escrito por extenso ou em numeral).
      - **REGRA DE CONTAGEM DEFINITIVA:** A primeira contagem explícita encontrada (seja por `SUBSTITUIR=X` ou "TRAZER X...") define o número final. Descrições detalhadas que vêm depois (ex: "um poste de 12m e outro de 14m") NÃO devem ser somadas ao total já definido.
      - Se nenhuma contagem explícita for encontrada, mas a troca do item for mencionada (ex: "trocar poste"), a contagem é 1.

    * **Para Cabos e Outros:**
      -Concidere jumpers igual a cabos. a contagem maxima deve ser **1**, somente para identificar de maneira geral a necessidade de intervenção
      - Se nenhuma avaria de prioridade maior for encontrada, identifique outras tarefas principais (ex: "trocar cruzeta", "poda de árvore").
      - A contagem para "Outros" deve ser **1**, representando a necessidade de uma intervenção geral, mesmo que haja mais de uma tarefa menor.

    ---
    ### 3. FORMATO DE SAÍDA OBRIGATÓRIO (JSON)
    A saída deve ser um objeto JSON válido. **Apenas UM campo pode ter valor maior que zero.** Os outros três devem ser 0.

    ---
    ### 4. EXEMPLOS PRÁTICOS

    **Exemplo 1 (Prioridade: Poste, Contagem: `SUBSTITUIR=X`)**
    * Texto da OS: "POSTE AVARIADO/ABALROADO (SUBSTITUIR=2 PS=1106031697), NECESSARIO LEVAR PS, SUBSTITUIR CRUZETA(S=5)"
    * Raciocínio: Prioridade máxima é Poste. Contagem explícita `SUBSTITUIR=2`. Ignora todo o resto.
    * Saída JSON:
      ```json
      {{
          "Postes": 2,
          "Trafos": 0,
          "Cabos": 0,
          "Outros": 0
      }}
      ```

    **Exemplo 2 (Prioridade: Trafo, Contagem: Implícita)**
    * Texto da OS: "TR DANIFICADO NA CHAVE. FAZER A SUBSTITUICAO DO TRAFO. CABO DO VIZINHO ROMPIDO."
    * Raciocínio: Sem Postes. Prioridade é Trafo. "SUBSTITUICAO DO TRAFO" implica contagem 1. Ignora o cabo.
    * Saída JSON:
      ```json
      {{
          "Postes": 0,
          "Trafos": 1,
          "Cabos": 0,
          "Outros": 0
      }}
      ```

    **Exemplo 3 (Prioridade: Poste, Contagem: Frase)**
    * Texto da OS: "TRAZER DOIS POSTE UM B2000 COMPLETO ESTRUTURA N4DN3, TRAZER UM B600 ESTRUTURA N3DN3CF"
    * Raciocínio: Prioridade máxima é Poste. A frase "TRAZER DOIS POSTE" define a contagem final como 2. O resto é descrição.
    * Saída JSON:
      ```json
      {{
          "Postes": 2,
          "Trafos": 0,
          "Cabos": 0,
          "Outros": 0
      }}
      ```

    **Exemplo 4 (Prioridade: Outros)**
    * Texto da OS: "Equipe no local identificou necessidade de PODA DE ARVORE proximo a rede e troca de uma cruzeta."
    * Raciocínio: Sem Postes, Trafos ou Cabos. Há outras tarefas. A contagem para o evento "Outros" é 1.
    * Saída JSON:
      ```json
      {{
          "Postes": 0,
          "Trafos": 0,
          "Cabos": 0,
          "Outros": 1
      }}
      ```

    ---
    ### TAREFA
    Analise o "Texto da NDS" abaixo seguindo rigorosamente todas as regras e exemplos. Retorne apenas o objeto JSON final.

    **Texto da OS:**
    "{description}"
    """
    return prompt_final

def analyze_with_gemini(description, model):
    """Envia o texto para a IA e retorna a contagem."""
    # (O conteúdo desta função permanece exatamente o mesmo)
    if not isinstance(description, str) or not description.strip():
        return (0, 0, 0, 0)
    prompt = build_gemini_prompt(description)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_response)
        return (data.get("Postes", 0), data.get("Trafos", 0), data.get("Cabos", 0), data.get("Outros", 0))
    except Exception as e:
        return f"Erro na análise: {e}"

# --- Interface do Utilizador ---

st.title("Analisador Inteligente de Nota de Serviço")
st.markdown("Faça o upload da sua planilha CSV para que a IA possa analisar e quantificar as avarias.")

uploaded_file = st.file_uploader(
    "Escolha um ficheiro CSV", 
    type="csv",
    help="O ficheiro deve ser separado por ponto e vírgula (;) e conter a coluna 'DESCRICAO DO SERVICO'."
)

if uploaded_file is not None:
    st.info(f"Ficheiro carregado: **{uploaded_file.name}**. Clique no botão abaixo para iniciar a análise.")

    if st.button("Analisar Ficheiro", type="primary"):
        model = carregar_modelo_ia()
        
        if model:
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
                
                coluna_descricao = 'DESCRICAO DO SERVICO'
                colunas_limpas = [col.strip() for col in df.columns]
                
                if coluna_descricao not in colunas_limpas:
                    st.error(f"Erro: A coluna obrigatória '{coluna_descricao}' não foi encontrada no ficheiro.")
                    st.write("Colunas encontradas:", df.columns.tolist())
                else:
                    # ### INÍCIO DA LÓGICA DE LOTES ###
                    
                    TAMANHO_DO_LOTE = 5  # Processa 3 pedidos de cada vez
                    PAUSA_ENTRE_LOTES = 5 # Espera 5 segundos entre cada lote

                    descriptions = df[coluna_descricao].tolist()
                    total_rows = len(descriptions)
                    results = []
                    
                    progress_bar = st.progress(0, text="A iniciar análise em lotes...")

                    # Divide a lista de descrições em lotes
                    for i in range(0, total_rows, TAMANHO_DO_LOTE):
                        lote_descriptions = descriptions[i:i + TAMANHO_DO_LOTE]
                        
                        # Processa um lote em paralelo
                        with concurrent.futures.ThreadPoolExecutor(max_workers=TAMANHO_DO_LOTE) as executor:
                            # Mapeia as descrições do lote para a função de análise
                            lote_results = list(executor.map(lambda desc: analyze_with_gemini(desc, model), lote_descriptions))
                            
                            # Adiciona os resultados do lote à lista principal
                            for result in lote_results:
                                if isinstance(result, str) and "Erro" in result:
                                    st.warning(f"Uma linha falhou com o erro: {result}")
                                    results.append((0, 0, 0, 0)) # Adiciona zeros em caso de erro
                                else:
                                    results.append(result)

                        # Atualiza a interface
                        linhas_processadas = len(results)
                        progress_value = linhas_processadas / total_rows
                        progress_bar.progress(progress_value, text=f"Lote concluído. {linhas_processadas}/{total_rows} linhas processadas.")

                        # Faz a pausa entre os lotes para não exceder o limite
                        if i + TAMANHO_DO_LOTE < total_rows:
                            st.write(f"A aguardar {PAUSA_ENTRE_LOTES} segundos para respeitar o limite da API...")
                            time.sleep(PAUSA_ENTRE_LOTES)

                    progress_bar.progress(1.0, text="Análise concluída!")
                    # ### FIM DA LÓGICA DE LOTES ###
                    
                    df[['Postes', 'Trafos', 'Cabos', 'Outros']] = pd.DataFrame(results, index=df.index)

                    st.success("Análise concluída com sucesso!")
                    st.markdown("### Pré-visualização dos Resultados")
                    st.dataframe(df)

                    @st.cache_data
                    def convert_df_to_csv(df_to_convert):
                        return df_to_convert.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')

                    csv_final = convert_df_to_csv(df)
                    
                    st.download_button(
                        label="📥 Baixar Ficheiro CSV com os Resultados",
                        data=csv_final,
                        file_name=f"resultado_{uploaded_file.name}",
                        mime='text/csv',
                    )

            except Exception as e:
                st.error(f"Ocorreu um erro ao ler ou processar o ficheiro CSV: {e}")