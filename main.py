import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool, FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Validações básicas (mantive as tuas correções)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("ERRO CRÍTICO: Falta a GOOGLE_API_KEY no ficheiro .env.")

warnings.filterwarnings('ignore')

llm_groq = LLM(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1 # Aumentei ligeiramente para ele ser criativo na busca, mas pouco
)

class SearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Useful to search for lists of startups, software houses and tech companies in specific locations."
    
    def _run(self, query: str) -> str:
        try:
            # Forçamos a pesquisa a excluir grandes agregadores para evitar ruído
            modified_query = f"{query} -site:indeed.com -site:linkedin.com/jobs"
            search_engine = DuckDuckGoSearchRun()
            return search_engine.run(modified_query)
        except Exception as e:
            return f"Error searching: {str(e)}"

scrape_tool = ScrapeWebsiteTool()
text_to_pdf_tool = FileReadTool()
search_tool = SearchTool()

# --- AGENTES REDEFINIDOS ---

# 1. O Perfilador: Não apenas lê, ele VENDE.
recolhedor = Agent(
    role="Tech Talent Profiler",
    goal="Extract specialized skills and projects that make the candidate attractive to AI Startups.",
    backstory="You are an expert tech recruiter. You don't care about generic soft skills. "
              "You look for hard tech stacks (Python, React, SQL), specific algorithms (Backtracking, AI models), "
              "and evidence of autonomy (projects, hackathons). You treat the CV as a sales pitch.",
    llm=llm_groq,
    verbose=True,
    max_iter=2
)

# 2. O Caçador de Startups: Focado em PMEs e contacto direto.
pesquisador = Agent(
    role="Startup Scout & Lead Generator",
    goal="Identify a list of high-potential small tech companies (<50 employees) in the target location.",
    verbose=True,
    backstory="You specialize in finding hidden gems: software boutiques, AI startups, and spin-offs in Switzerland. "
              "You ignore large corporations. You focus on finding the 'Careers' page or the 'About Us' page to find "
              "direct contact emails (CTO, Founders, HR) for spontaneous applications.",
    llm=llm_groq, 
    max_iter=4 # Mais iterações porque pesquisar é difícil
)

# --- TAREFAS OTIMIZADAS ---

# Tarefa 1: Extrair o "Ouro" do teu CV
recolhedor_tarefa = Task(
    description=(
        "Analyze the user's CV at '{cv}'. \n"
        "Identify the top 3 'Killer Projects' and the core Tech Stack.\n"
        "Create a 'Value Proposition' summary: Why should a Swiss CEO hire this 18-year-old? "
        "Focus on the combination of Data Science + Full Stack skills."
    ),
    expected_output="A structured summary containing: Top Skills, Key Projects, and a 2-sentence Elevator Pitch.",
    agent=recolhedor,
    tools=[text_to_pdf_tool]
)

# Tarefa 2: Encontrar os Alvos
pesquisar_tarefa = Task(
    description=(
        "Based on the user's skills (from the previous task), find 5-10 companies in {place} that match these criteria:\n"
        "1. **Size:** Small to Medium (10-50 employees). STRICTLY NO LARGE CORPORATIONS.\n"
        "2. **Sector:** AI, Fintech, MedTech, or Software Agencies (SaaS).\n"
        "3. **Relevance:** They must use technologies similar to the user's stack (Python, React, Data).\n"
        "4. **Actionability:** Look for 'Team', 'About', or 'Contact' pages to find a specific email or contact form.\n\n"
        "Use search queries like: 'AI startups Zurich', 'Software boutiques Zurich', 'Swiss TechnoPark companies list', 'Top swiss startups 2025'.\n"
        "Do not look for 'internship ads'. Look for *companies* that do interesting work."
    ),
    expected_output=(
        "A Markdown list of 5-10 companies. Each item must have:\n"
        "- **Company Name**\n"
        "- **Why it's a match** (Connect their product to user's skills)\n"
        "- **Website URL**\n"
        "- **Target Contact** (Generic email or specific founder name if found)\n"
    ),
    agent=pesquisador,
    context=[recolhedor_tarefa],
    tools=[search_tool, scrape_tool]
)

job_searcher_crew = Crew(
    agents=[recolhedor, pesquisador],
    tasks=[recolhedor_tarefa, pesquisar_tarefa],
    verbose=True,
    process='sequential'
)

# Ajustei o objetivo para ser menos "pedinte" e mais estratégico
user_detalhes = {
    "cv": "./cv_afonso.md",
    "place": "Zurich, Switzerland", 
    # Removi o "user_goal" daqui porque embuti a lógica nas Tasks para serem mais diretivas
}

print("Iniciando a caça às startups...")
res = job_searcher_crew.kickoff(inputs=user_detalhes)
print("\n--- RELATÓRIO DE ALVOS ---")
print(res)