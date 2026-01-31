import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool, FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

if not os.getenv("GROQ_API_KEY2"):
    raise ValueError("ERRO: Falta a GROQ_API_KEY no .env")

warnings.filterwarnings('ignore')

# Usamos o Mixtral aqui se possível, ou o Llama. O Mixtral é melhor a seguir listas.
llm_groq = LLM(
    api_key=os.getenv("GROQ_API_KEY3"),
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1
)

# --- TOOLS ---
class SearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search for specific lists of startups."
    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)

scrape_tool = ScrapeWebsiteTool()
text_to_pdf_tool = FileReadTool()
search_tool = SearchTool()

# --- AGENTES ---

recolhedor = Agent(
    role="Tech Talent Profiler",
    goal="Extract specialized skills.",
    backstory="You extract hard skills from CVs.",
    llm=llm_groq,
    verbose=True
)

pesquisador = Agent(
    role="Spinoff Scout",
    goal="Find 5 ETH Zurich Spinoffs or Robotics Startups.",
    backstory="You specialize in finding early-stage deep-tech companies in Zurich. "
              "You look for companies founded in the last 5 years. "
              "You prioritize finding an Email, but if you can't, you ALWAYS keep the Website URL.",
    llm=llm_groq,
    verbose=True,
    max_iter=5
)

revisor = Agent(
    role="Strategic Filter",
    goal="Filter out hardware and non-tech companies.",
    backstory="You are ruthless. If a company sells physical products (drones, batteries, pills), you DELETE it. "
              "You only keep companies where a Python/JS/SQL developer is the core asset. "
              "You ensure the company is small enough that a cold email might actually reach the CEO.",
    llm=llm_groq,
    verbose=True
)
# --- TAREFAS ---

task_cv = Task(
    description="Analyze the CV at '{cv}'. Identify top 3 hard skills.",
    expected_output="Summary of skills.",
    agent=recolhedor,
    tools=[text_to_pdf_tool]
)

task_search = Task(
    description=(
        "Find 5 **Software/SaaS Startups** specifically in **Zurich (Canton or City)**.\n"
        "**STRICT LOCATION RULE:** The company HQ MUST be in Zurich. Ignore Geneva, Lausanne, or Germany.\n\n"
        "**SEARCH STRATEGY:**\n"
        "1. Search for: 'Zurich AI startups list 2024', 'SaaS companies based in Zurich', 'Top Fintech Zurich seed stage'.\n"
        "2. Focus on B2B Software, Fintech, or LegalTech. (Exclude Hardware/Biotech).\n"
        "3. **EMAIL HUNTING:** For each company, specifically search '[Company Name] contact email' or '[Company Name] careers email'.\n"
        "4. **EMAIL EXTRACTION:** You MUST copy the exact email address if found on the website.\n"
        "5. If you cannot find an email on the page, write specifically: 'EMAIL_NOT_FOUND'. Do not leave it blank.\n"
        "SEARCH TRICK: For Swiss companies, search for 'Company Name Impressum' or 'Company Name Privacy Policy'. The email is often hidden there legally."
    ),
    expected_output="A raw list of 5 companies with: Name | Location (Must be Zurich) | Contact (Email OR 'NO_EMAIL_FOUND') | Tech Stack/Niche.",
    agent=pesquisador,
    context=[task_cv],
    tools=[search_tool, scrape_tool]
)
task_review = Task(
    description=(
        "Review the Scout's list.\n"
        "1. Filter out any company that looks like a giant corporation.\n"
        "2. Format the final output strictly as a pipe-separated list:\n"
        "**Name | Contact (Email OR Link) | Why it fits**\n\n"
        "Example output:\n"
        "Anybotics | info@anybotics.com | Robotics spinoff using AI.\n"
        "LatticeFlow | https://latticeflow.ai/contact | AI safety startup.\n"
    ),
    expected_output="A structured list of 5 real startups with contacts.",
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

print("### A CAÇAR SPINOFFS (LISTA GARANTIDA) ###")
resultado_final = crew.kickoff(inputs=inputs)

# --- GUARDAR ---
nome_ficheiro = "lista_empresas.txt"
with open(nome_ficheiro, "w", encoding="utf-8") as f:
    f.write(str(resultado_final))

print(f"\n✅ Lista guardada em '{nome_ficheiro}'.")
print("Podes correr o 'cold_message_refiner.py' agora.")