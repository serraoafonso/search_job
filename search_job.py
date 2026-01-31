import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool, FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Padronização da API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY3")
if not GROQ_API_KEY:
    raise ValueError("ERRO: Falta a GROQ_API_KEY no .env")

warnings.filterwarnings('ignore')

# Modelo Llama 3.3 70B via Groq
llm_groq = LLM(
    api_key=GROQ_API_KEY,
    model="groq/llama-3.3-70b-versatile",
    temperature=0.0 
)

# --- TOOLS ---
class SearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search for specific lists of startups. Use specific queries."
    def _run(self, query: str) -> str:
        clean_query = f"{query} -site:indeed.com -site:linkedin.com/jobs -site:glassdoor.com"
        return DuckDuckGoSearchRun().run(clean_query)

scrape_tool = ScrapeWebsiteTool()
file_read_tool = FileReadTool()
search_tool = SearchTool()

# --- AGENTES ---

recolhedor = Agent(
    role="Tech Talent Profiler",
    goal="Extract real technical skills and projects from the CV.",
    backstory="You are a technical recruiter focused on hard skills. You ignore generic soft skills and focus on what the candidate can build. You communicate in English.",
    llm=llm_groq,
    verbose=True
)

pesquisador = Agent(
    role="Spinoff & Startup Scout",
    goal="Find 5 real startups in {place} that match the candidate's stack.",
    backstory="""You are an expert in finding early-stage tech companies. 
    You focus on Deep Tech, SaaS, and AI. 
    CRITICAL EMAIL RULE: Only report an email if you explicitly find it. 
    NEVER invent or assume an email. 
    If you don't find one, write exactly 'EMAIL_NOT_FOUND'. 
    Search in 'Contact', 'About Us', or 'Impressum' pages.
    All your output must be in English.""",
    llm=llm_groq,
    verbose=True,
    max_iter=5
)

revisor = Agent(
    role="Strategic Quality Controller",
    goal="Validate the company list and ensure data is real and in English.",
    backstory="""You are ruthless. You verify if the found companies are truly tech startups. 
    If you see an email that looks generic or made up without evidence, mark it as 'EMAIL_NOT_FOUND'.
    Ensure the final format is strictly followed and everything is in English.""",
    llm=llm_groq,
    verbose=True
)

# --- TAREFAS ---

task_cv = Task(
    description="Analyze the CV at '{cv}'. Identify the top 3 technologies and 2 relevant projects.",
    expected_output="Structured summary of skills and projects in English.",
    agent=recolhedor,
    tools=[file_read_tool]
)

task_search = Task(
    description=(
        "Find 5 Software/SaaS Startups based in {place}.\n"
        "**STRATEGY:**\n"
        "1. Search for recent startup lists (2024/2025) in {place}.\n"
        "2. For each company, try to find the official website.\n"
        "3. **EMAIL DETECTION:** Try to find a real email. If the site has an 'Impressum' page, the email is there.\n"
        "4. **NO HALLUCINATION:** If you don't see an email with your own 'eyes' (scrape tool), write 'EMAIL_NOT_FOUND'.\n"
    ),
    expected_output="A list of 5 companies with: Name | Website | Email (or EMAIL_NOT_FOUND) | Why it is a match. All in English.",
    agent=pesquisador,
    context=[task_cv],
    tools=[search_tool, scrape_tool]
)

task_review = Task(
    description=(
        "Review the Scout's list.\n"
        "1. Remove companies that are not purely tech-focused.\n"
        "2. Confirm if emails look legitimate. If they look 'guessed' (e.g., contact@company.com without scraping), change to EMAIL_NOT_FOUND.\n"
        "3. Format as a pipe-separated list (|).\n"
        "Format: Name | Email or Contact Link | Short Description"
    ),
    expected_output="Final formatted list of 5 startups with validated contacts in English.",
    agent=revisor,
    context=[task_search]
)

# --- EQUIPA ---
crew = Crew(
    agents=[recolhedor, pesquisador, revisor],
    tasks=[task_cv, task_search, task_review],
    verbose=True
)

inputs = {"cv": "./cv_afonso.md", "place": "Zurich, Switzerland"}

if __name__ == "__main__":
    print("### STARTING STARTUP SEARCH (FOCUS ON REAL DATA & ENGLISH) ###")
    resultado_final = crew.kickoff(inputs=inputs)

    nome_ficheiro = "lista_empresas.txt"
    with open(nome_ficheiro, "w", encoding="utf-8") as f:
        f.write(str(resultado_final))

    print(f"\n✅ List saved to '{nome_ficheiro}'.")
