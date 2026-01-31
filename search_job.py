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
    temperature=0.0 # Temperatura zero para reduzir alucinações
)

# --- TOOLS ---
class SearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search for specific lists of startups. Use specific queries."
    def _run(self, query: str) -> str:
        # Melhora a busca excluindo sites de emprego genéricos
        clean_query = f"{query} -site:indeed.com -site:linkedin.com/jobs -site:glassdoor.com"
        return DuckDuckGoSearchRun().run(clean_query)

scrape_tool = ScrapeWebsiteTool()
file_read_tool = FileReadTool()
search_tool = SearchTool()

# --- AGENTES ---

recolhedor = Agent(
    role="Tech Talent Profiler",
    goal="Extrair competências técnicas reais e projetos do CV.",
    backstory="És um recrutador técnico focado em 'hard skills'. Ignoras 'soft skills' genéricas e focas-te no que o candidato sabe construir.",
    llm=llm_groq,
    verbose=True
)

pesquisador = Agent(
    role="Spinoff & Startup Scout",
    goal="Encontrar 5 startups reais em {place} que usem o stack do candidato.",
    backstory="""És um especialista em encontrar empresas tecnológicas em fase inicial. 
    Focas-te em 'Deep Tech', 'SaaS' e 'AI'. 
    REGRA CRÍTICA PARA E-MAILS: Só reportas um e-mail se o encontrares explicitamente. 
    NUNCA inventes ou assumas um e-mail (ex: info@empresa.com). 
    Se não encontrares, escreve 'EMAIL_NOT_FOUND'. 
    Procura em páginas de 'Contact', 'About Us' ou 'Impressum'.""",
    llm=llm_groq,
    verbose=True,
    max_iter=5
)

revisor = Agent(
    role="Strategic Quality Controller",
    goal="Validar a lista de empresas e garantir que os dados são reais.",
    backstory="""És implacável. Verificas se as empresas encontradas são realmente startups tecnológicas. 
    Se vires um e-mail que pareça genérico ou inventado pelo agente anterior sem evidência, marcas como 'EMAIL_NOT_FOUND'.
    Garante que o formato final é seguido rigorosamente.""",
    llm=llm_groq,
    verbose=True
)

# --- TAREFAS ---

task_cv = Task(
    description="Analisa o CV em '{cv}'. Identifica as 3 principais tecnologias e 2 projetos relevantes.",
    expected_output="Resumo estruturado das competências e projetos.",
    agent=recolhedor,
    tools=[file_read_tool]
)

task_search = Task(
    description=(
        "Encontra 5 Startups de Software/SaaS sediadas em {place}.\n"
        "**ESTRATÉGIA:**\n"
        "1. Pesquisa por listas de startups recentes (2024/2025) em {place}.\n"
        "2. Para cada empresa, tenta encontrar o site oficial.\n"
        "3. **DETEÇÃO DE E-MAIL:** Tenta encontrar um e-mail real. Se o site tiver uma página 'Impressum', o e-mail está lá.\n"
        "4. **PROIBIDO ALUCINAR:** Se não vires um e-mail com os teus próprios 'olhos' (ferramenta de scrape), escreve 'EMAIL_NOT_FOUND'.\n"
    ),
    expected_output="Uma lista de 5 empresas com: Nome | Website | E-mail (ou EMAIL_NOT_FOUND) | Porquê é um match.",
    agent=pesquisador,
    context=[task_cv],
    tools=[search_tool, scrape_tool]
)

task_review = Task(
    description=(
        "Revê a lista do Scout.\n"
        "1. Remove empresas que não sejam puramente tecnológicas.\n"
        "2. Confirma se os e-mails parecem legítimos. Se parecerem 'chutados' (ex: contact@company.com sem scraping), altera para EMAIL_NOT_FOUND.\n"
        "3. Formata como uma lista separada por pipes (|).\n"
        "Formato: Nome | E-mail ou Link de Contacto | Descrição Curta"
    ),
    expected_output="Lista final formatada de 5 startups com contactos validados.",
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
    print("### INICIANDO PESQUISA DE STARTUPS (FOCO EM DADOS REAIS) ###")
    resultado_final = crew.kickoff(inputs=inputs)

    # --- GUARDAR ---
    nome_ficheiro = "lista_empresas.txt"
    with open(nome_ficheiro, "w", encoding="utf-8") as f:
        f.write(str(resultado_final))

    print(f"\n✅ Lista guardada em '{nome_ficheiro}'.")
