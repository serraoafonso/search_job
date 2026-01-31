import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import random
import re

load_dotenv()

# --- 1. CONFIGURAÇÃO ---
if not os.getenv("GROQ_API_KEY2"):
    raise ValueError("ERRO: Falta a GROQ_API_KEY no .env")

# Usamos o Llama 3 70B porque é ótimo a seguir instruções complexas
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY3"), 
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# --- 2. O ESTADO (A Memória do Agente) ---
class AgentState(TypedDict):
    lista_bruta: str   # O ficheiro txt completo que veio do outro script
    rascunho: str      # O email que está a ser escrito
    critica: str       # O feedback do revisor
    iteracoes: int     # Para não ficar num loop infinito

# --- 3. OS AGENTES (NÓS) ---
import random
import re

def escritor(state: AgentState):
    print(f"\n✍️  ESCRITOR A TRABALHAR (Tentativa {state['iteracoes'] + 1})...")
    
    # 1. PYTHON: Processar a lista bruta
    linhas = [l for l in state['lista_bruta'].split('\n') if l.strip()]
    
    if not linhas:
        return {"rascunho": "ERRO: Lista vazia.", "iteracoes": state['iteracoes']}

    # 2. PYTHON: Escolher uma empresa aleatória (Adeus viés de posição)
    empresa_alvo_texto = random.choice(linhas)
    
    # 3. PYTHON: Caçar o email com REGEX (O LLM não toca nisto)
    # Procura padrões de email reais
    match_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', empresa_alvo_texto)
    
    if match_email:
        email_final = match_email.group(0)
    else:
        email_final = "MISSING_EMAIL"

    print(f"   -> Alvo selecionado: {empresa_alvo_texto}")
    print(f"   -> Email detetado (Regex): {email_final}")

    # 4. LLM: Escrever o corpo (Agora ele é obrigado a usar o email que definimos)
    prompt = f"""
    TARGET COMPANY INFO:
    {empresa_alvo_texto}

    TARGET EMAIL (STRICT): {email_final}

    MY PROFILE:
    - Name: Afonso (18).
    - Availability: July 1st - Sept 30th, 2026.
    - Tech: Python (Automation/Agents), JS (Algorithms), SQL.
    - Proof: Built a Job Hunting Agent + Sudoku Solver (Backtracking in JS).
    - Stats: 18.4 Math GPA.

    INSTRUCTIONS:
    1. **Analyze the Industry:** Based strictly on the 'TARGET COMPANY INFO', determine if they are Fintech, AI, SaaS, GreenTech, etc.
    2. **Customize the Pain Point:**
       - If Fintech -> Offer SQL/Data Integrity.
       - If AI/ML -> Offer Python Scripts/Data Labelling.
       - If Web SaaS -> Offer React Components/Testing.
    3. **Write the Email:**
       - Use a unique opening hook based on their industry.
       - Mention "Grunt Work" (cleaning data, tests, dashboards).
       - Keep it short and punchy.

    STRICT OUTPUT FORMAT:
    To: {email_final}
    Subject: [Creative Subject about "Grunt Work" for their Industry]

    [Email Body]
    
    PREVIOUS CRITIQUE: {state.get('critica', 'None')}
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"rascunho": msg.content, "iteracoes": state['iteracoes'] + 1}

def critico(state: AgentState):
    print("\n🧐 CRÍTICO A VALIDAR ESTRATÉGIA...")
    
    prompt = f"""
    You are a Career Strategy Mentor. Review this cold email draft specifically for a Zurich tech startup:
    
    {state['rascunho']}
    
    VALIDATION CHECKLIST (Focus on the STRATEGY, not just grammar):

    1. **The "Grunt Work" Value Prop:** Does the email clearly convey that Afonso wants to do the boring tasks (data cleaning, QA, internal tools) to save the senior team time? (It doesn't need the exact words "grunt work", but the concept must be clear).
    2. **Logistics:** Are the dates (July-Sept 2026) clearly stated?
    3. **Tech Integrity:** Does it correctly attribute **Python** to the AI Agent project and **JavaScript** to the Sudoku/Algorithms project?
    4. **Safety Check:** Is the 'To:' field either a valid email with '@' OR exactly "MISSING_EMAIL"?

    DECISION RULES:
    - **PASS:** If the logic is sound and the offer is helpful, reply exactly: APROVADO.
    - **FAIL:** If the email sounds too arrogant (like he's teaching them) OR if he forgets to offer help with the boring tasks, tell him to fix the tone.
    - **FAIL:** If dates or tech stack are mixed up.
    
    Output your feedback clearly.
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"critica": msg.content}
# --- 4. O FLUXO (CONTROLO) ---
def verificar_qualidade(state: AgentState):
    if "APROVADO" in state['critica'] or state['iteracoes'] >= 3:
        return "fim"
    return "refazer"

# --- 5. MONTAGEM DO GRAFO ---
workflow = StateGraph(AgentState)
workflow.add_node("escritor", escritor)
workflow.add_node("critico", critico)

workflow.set_entry_point("escritor")
workflow.add_edge("escritor", "critico")
workflow.add_conditional_edges(
    "critico",
    verificar_qualidade,
    {
        "refazer": "escritor",
        "fim": END
    }
)

app = workflow.compile()

# --- 6. EXECUÇÃO ---

# Passo 1: Ler o ficheiro gerado pelo Script Anterior
arquivo_empresas = "lista_empresas.txt"
if not os.path.exists(arquivo_empresas):
    print(f"❌ ERRO: O ficheiro '{arquivo_empresas}' não existe.")
    print("👉 Corre primeiro o 'job_hunter.py' para encontrar as empresas!")
    exit()

    
with open(arquivo_empresas, "r", encoding="utf-8") as f:
    linhas = f.readlines()

# TRUQUE ANTI-REPETIÇÃO:
# 1. Baralha as linhas para o LLM não viciar na primeira.
# 2. (Opcional) Podes filtrar linhas vazias.
random.shuffle(linhas) 
conteudo_lista = "".join(linhas)

# Passo 2: Iniciar o Agente
inputs = {
    "lista_bruta": conteudo_lista,
    "rascunho": "",
    "iteracoes": 0
}

print("### A SELECIONAR O ALVO E A ESCREVER O EMAIL ###")
res = app.invoke(inputs)

print("\n" + "="*50)
print("📧 MENSAGEM FINAL (COPIA E ENVIA AGORA)")
print("="*50)
print(res['rascunho'])
print("="*50)