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
    print(f"\n✍️  WRITER AT WORK (Attempt {state['iteracoes'] + 1})...")
    
    linhas = [l.strip() for l in state['lista_bruta'].split('\n') if l.strip() and '|' in l]
    if not linhas:
        return {"rascunho": "ERROR: Empty or invalid list.", "iteracoes": state['iteracoes']}

    # Escolher uma empresa aleatória
    empresa_alvo_texto = random.choice(linhas)
    
    # Extração robusta de e-mail usando Regex
    match_email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', empresa_alvo_texto)
    
    if not match_email:
        # SE NÃO HOUVER EMAIL, PARAMOS AQUI
        msg_erro = f"❌ SKIPPING: No email found for target: {empresa_alvo_texto.split('|')[0].strip()}"
        print(msg_erro)
        return {"rascunho": msg_erro, "iteracoes": 3} # Forçamos o fim do loop

    email_final = match_email.group(0)
    print(f"   -> Target selected: {empresa_alvo_texto}")
    print(f"   -> Email detected: {email_final}")

    prompt = f"""
    TARGET COMPANY INFO:
    {empresa_alvo_texto}

    TARGET EMAIL (MANDATORY): {email_final}

    CANDIDATE PROFILE:
    - Name: Afonso (18 years old).
    - Availability: July 1st to September 30th, 2026.
    - Stack: Python (Automation/Agents), JS (Algorithms), SQL.
    - Projects: Job Hunting Agent (CrewAI/LangGraph), Sudoku Solver (Backtracking).
    - Academic: 18.4 Math GPA.

    INSTRUCTIONS:
    1. **Language:** The entire email MUST be in English.
    2. **Personalization:** Identify the company sector (AI, Fintech, etc.).
    3. **Value Prop:** Offer to do the "grunt work" (data cleaning, testing, dashboards).
    4. **Tone:** Professional, humble, and direct.
    5. **Format:**
       To: {email_final}
       Subject: [Creative Subject about helping with {email_final}]
       
       [Email Body in English]

    PREVIOUS CRITIQUE: {state.get('critica', 'None')}
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"rascunho": msg.content, "iteracoes": state['iteracoes'] + 1}

def critico(state: AgentState):
    # Se já houve um erro de "No email found", não precisamos criticar
    if "SKIPPING" in state['rascunho']:
        return {"critica": "APROVADO"}

    print("🧐 CRITIC VALIDATING...")
    
    prompt = f"""
    You are a Career Mentor. Review this cold email draft for a startup in Zurich:
    
    {state['rascunho']}
    
    CHECKLIST:
    1. Is the email in English? (Mandatory)
    2. Does it mention availability (July-Sept 2026)?
    3. Is the value prop clear (helping with grunt work)?
    4. Is the 'To:' field correct and not hallucinated?

    If it's perfect, reply only: APPROVED.
    Otherwise, provide correction instructions in English.
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"critica": msg.content}

def verificar_qualidade(state: AgentState):
    if "APPROVED" in state['critica'] or "APROVADO" in state['critica'] or state['iteracoes'] >= 3:
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
        print(f"❌ ERROR: '{arquivo_empresas}' not found. Run 'search_job.py' first.")
        exit()
        
    with open(arquivo_empresas, "r", encoding="utf-8") as f:
        conteudo = f.read()

    print("### GENERATING REFINED COLD MESSAGE (STRICT ENGLISH) ###")
    res = app.invoke({"lista_bruta": conteudo, "rascunho": "", "iteracoes": 0})

    print("\n" + "="*50)
    print("📧 FINAL OUTPUT")
    print("="*50)
    print(res['rascunho'])
    print("="*50)
