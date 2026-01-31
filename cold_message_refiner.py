import os
import random
import re
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# Padronização da API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY3")
if not GROQ_API_KEY:
    raise ValueError("ERRO: Falta a GROQ_API_KEY no .env")

# Configuração do LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY, 
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# --- ESTADO ---
class AgentState(TypedDict):
    lista_bruta: str   
    rascunho: str      
    critica: str       
    iteracoes: int     

# --- AGENTES ---

def escritor(state: AgentState):
    print(f"\n✍️  ESCRITOR A TRABALHAR (Tentativa {state['iteracoes'] + 1})...")
    
    linhas = [l.strip() for l in state['lista_bruta'].split('\n') if l.strip()]
    if not linhas:
        return {"rascunho": "ERRO: Lista vazia.", "iteracoes": state['iteracoes']}

    # Escolher uma empresa aleatória
    empresa_alvo_texto = random.choice(linhas)
    
    # Extração robusta de e-mail usando Regex
    # Procura por algo que pareça um email, mas ignora se for apenas um link de contato
    match_email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', empresa_alvo_texto)
    
    if match_email:
        email_final = match_email.group(0)
    else:
        # Se não houver email, tentamos extrair um URL de contato
        match_url = re.search(r'https?://[^\s|]+', empresa_alvo_texto)
        email_final = match_url.group(0) if match_url else "CONTACT_FORM_REQUIRED"

    print(f"   -> Alvo selecionado: {empresa_alvo_texto}")
    print(f"   -> Contacto detetado: {email_final}")

    prompt = f"""
    INFO DA EMPRESA ALVO:
    {empresa_alvo_texto}

    CONTACTO (OBRIGATÓRIO): {email_final}

    PERFIL DO CANDIDATO:
    - Nome: Afonso (18 anos).
    - Disponibilidade: 1 de Julho a 30 de Setembro de 2026.
    - Stack: Python (Automação/Agentes), JS (Algoritmos), SQL.
    - Projetos: Job Hunting Agent (CrewAI/LangGraph), Sudoku Solver (Backtracking).
    - Académico: Média de 18.4 em Matemática.

    INSTRUÇÕES:
    1. **Personalização:** Identifica o setor da empresa (AI, Fintech, etc.).
    2. **Proposta de Valor:** Oferece-te para fazer o "trabalho pesado" (limpeza de dados, testes, dashboards).
    3. **Tom:** Profissional, mas humilde e direto.
    4. **Formato:**
       Para: {email_final}
       Assunto: [Assunto Criativo sobre ajuda em {email_final}]
       
       [Corpo do Email em Inglês, pois é para Zurique]

    CRÍTICA ANTERIOR: {state.get('critica', 'Nenhuma')}
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"rascunho": msg.content, "iteracoes": state['iteracoes'] + 1}

def critico(state: AgentState):
    print("🧐 CRÍTICO A VALIDAR...")
    
    prompt = f"""
    És um Mentor de Carreira. Revê este rascunho de cold email para uma startup em Zurique:
    
    {state['rascunho']}
    
    CHECKLIST:
    1. O e-mail está em Inglês? (Obrigatório para Zurique)
    2. Menciona a disponibilidade (Julho-Setembro 2026)?
    3. A proposta de valor é clara (ajudar com tarefas chatas/grunt work)?
    4. O contacto 'To:' está correto e não é inventado?

    Se estiver perfeito, responde apenas: APROVADO.
    Caso contrário, dá instruções de correção.
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"critica": msg.content}

def verificar_qualidade(state: AgentState):
    if "APROVADO" in state['critica'] or state['iteracoes'] >= 3:
        return "fim"
    return "refazer"

# --- GRAFO ---
workflow = StateGraph(AgentState)
workflow.add_node("escritor", escritor)
workflow.add_node("critico", critico)
workflow.set_entry_point("escritor")
workflow.add_edge("escritor", "critico")
workflow.add_conditional_edges("critico", verificar_qualidade, {"refazer": "escritor", "fim": END})

app = workflow.compile()

if __name__ == "__main__":
    arquivo_empresas = "lista_empresas.txt"
    if not os.path.exists(arquivo_empresas):
        print(f"❌ ERRO: '{arquivo_empresas}' não encontrado. Corre primeiro o 'search_job.py'.")
        exit()
        
    with open(arquivo_empresas, "r", encoding="utf-8") as f:
        conteudo = f.read()

    print("### GERANDO COLD MESSAGE REFINADA ###")
    res = app.invoke({"lista_bruta": conteudo, "rascunho": "", "iteracoes": 0})

    print("\n" + "="*50)
    print("📧 MENSAGEM FINAL")
    print("="*50)
    print(res['rascunho'])
    print("="*50)
