from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.website import WebsiteTools
from dotenv import load_dotenv
from phi.llm.azure import AzureOpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from pathlib import Path
from phi.knowledge.text import TextKnowledgeBase
from phi.embedder.azure_openai import AzureOpenAIEmbedder
from phi.vectordb.pgvector import PgVector2

load_dotenv()

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

azure_embedder = AzureOpenAIEmbedder(
    api_key="azure openai key, i have removed of secruity phases",
    azure_deployment="text-embedding-3-small",
    azure_endpoint="https://jobspringai.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
)

# Initialize the TextKnowledgeBase
knowledge_base = TextKnowledgeBase(
    path=Path("text_documents"),  # Table name: ai.text_documents
    vector_db=PgVector2(
        collection="text_documents",
        db_url=db_url,
        embedder=azure_embedder,
    ),
    num_documents=5,  # Number of documents to return on search
)
# Load the knowledge base
knowledge_base.load(recreate=False)

searcher = Assistant(
    llm=AzureOpenAIChat(model="gpt-4"),
    monitoring=True,
    name="Researcher",
    role="Tech Job Researcher from URLS",
    show_tool_calls=True,
    description=dedent(
        """\
      "Analyze the job posting URL provided URL "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements.".
    """
    ),
    instructions=[
         "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences.",
    ],
    tools=[WebsiteTools(),DuckDuckGo()],
    add_datetime_to_instructions=True,
)

resume_maker = Assistant(
    llm=AzureOpenAIChat(model="gpt-4"),
    knowledge_base=knowledge_base,
    add_references_to_prompt=True,
    tools=[DuckDuckGo()],
    name="resume maker",
    role="You are an expert tech job researcher specializing in detailed analysis of job postings to assist job applicants in crafting standout resumes.",
    show_tool_calls=True,
    description=dedent(
        """\
      " Deep Analysis: Navigate through job descriptions and extract critical details, including necessary qualifications, skills, and keywords that employers value most"
      "use the knowledge base to get student resume details for customization "
      "Use your expertise to pinpoint ways a resume can align with job requirements, ensuring it effectively highlights the applicant's strengths and relevance."
      "make a custom resume according to job description to provided resume text file ,make the resume stand out in a competitive job market.,"
      " modify to the applicant's experience, skills, and projects to match the job posting's tone, priorities, and expectations."
      " make sure the resume is ATS Friendly , that the resume should get very high score"
      "use required tools"
    """
    ),
    instructions=[
        "deliver a fully customized, ATS Friendly resume that aligns with the job description, highlighting the candidate's strengths and qualifications effectively."
        "make sure the resume is ATS Friendly , that the resume should get very high score",
    ],

)

editor = Assistant(
    llm=AzureOpenAIChat(model="gpt-4"),
    team=[searcher, resume_maker],
    description="ypu are a senior resume maker , your goal is edit the resume and the resume standout in the job market",
    instructions=[
        "get all the details from the job description from the given url from the searcher agent "
        "now from the resume_maker agent craft the resume according to the job description , use necessary tools  "
        "Remember: you are the final gatekeeper before the resume is used.",
    ]

)

editor.print_response("create a customized resume according to this url https://www.linkedin.com/jobs/view/4117464318/?alternateChannel=search&refId=wCIRXUUnGZOp1%2BZ%2FQ3tzKA%3D%3D&trackingId=bGd7eHye1OMkEDAvRtMv2Q%3D%3D")

