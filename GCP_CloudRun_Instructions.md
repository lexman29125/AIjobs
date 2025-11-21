# Deployment Instructions for Google Cloud Run
Both the backend and frontend components can be deployed independently to Google Cloud Run, a fully managed compute platform for stateless containers.

1. Backend Service Deployment (e.g., as fastapi-backend)
To deploy the backend, you'll typically wrap the core logic in a web framework like FastAPI and containerize it.

backend_main.py (Example FastAPI application):

# Save this content as backend_main.py
import os
import io
import PyPDF2
import asyncio
import nest_asyncio
import random
import re
import json
from google import generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# --- API Key Handling ---
load_dotenv() # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable.")

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

llm_model_name = "gemini-2.5-flash"

# Helper function to extract text from URL (same as in backend_logic.py)
def extract_text_from_url(url: str) -> str:
    # ... (same as the extract_text_from_url function in the backend code cell)
    try:
        headers = {
            'User-Agent': 'Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

def analyze_skills_and_gaps(resume_text: str, job_description_text: str) -> str:
    # ... (same as the analyze_skills_and_gaps function in the backend code cell)
    full_prompt = f"""
    You are an expert HR analyst. Your task is to compare a candidate's resume with a job description.
    Provide your output as a JSON object ONLY. Do not include any other text or explanation outside the JSON.

    Here is the candidate's Resume:
    ---
    {resume_text}
    ---

    Here is the Job Description:
    ---
    {job_description_text}
    ---

    JSON Schema:
    {{
        "candidate_skills": ["string"], # List of key technical and soft skills explicitly mentioned in the resume.
        "required_job_skills": ["string"], # List of essential technical and soft skills mentioned in the job description.
        "matched_skills": ["string"], # Skills present in both the resume and the job description.
        "missing_skills": ["string"], # Skills required by the job description but NOT found in the resume.
        "additional_skills": ["string"], # Skills present in the resume but not explicitly required by the job description.
        "overall_fit_summary": "string" # A brief summary of how well the candidate's skills align with the job requirements.
    }}
    """

    try:
        model = genai.GenerativeModel(llm_model_name)
        response = model.generate_content(contents=full_prompt)
        return response.text
    except Exception as e:
        return f"Error during LLM analysis: {e}"

def analyze_resume_job_description_full(resume_text: str, job_description_text: str) -> dict:
    # ... (same as the analyze_resume_job_description_full function in the backend code cell)
    print(f"Initiating LLM-based analysis for resume (length: {len(resume_text)}) and job description (length: {len(job_description_text)}).")
    analysis_report = analyze_skills_and_gaps(resume_text, job_description_text)

    if "Error during LLM analysis" in analysis_report:
        return {"analysis_status": "failure", "message": analysis_report}
    else:
        try:
            cleaned_report = analysis_report.strip()
            if cleaned_report.startswith('```json') and cleaned_report.endswith('```'):
                cleaned_report = cleaned_report[len('```json'):-len('```')].strip()

            parsed_report = json.loads(cleaned_report)
            return {"analysis_status": "success", "message": "LLM-based analysis completed and parsed.", "parsed_report": parsed_report}
        except json.JSONDecodeError as e:
            return {"analysis_status": "failure", "message": f"Failed to parse LLM output as JSON: {e}", "raw_report": analysis_report}
        except Exception as e:
            return {"analysis_status": "failure", "message": f"An unexpected error occurred during JSON parsing: {e}", "raw_report": analysis_report}


# Tool and Agent class definitions (same as in backend code cell)
class Tool:
    def __init__(self, func, name, description):
        self.func = func
        self.name = name
        self.description = description

class Agent:
    def __init__(self, name, instruction, tools: list):
        self.name = name
        self.instruction = instruction
        self.tools = tools

class CoordinatorAgent(Agent):
    def __init__(self, name: str, instruction: str, tools: list = None, sub_agents: list = None):
        super().__init__(name, instruction, tools if tools is not None else [])
        self.sub_agents = sub_agents if sub_agents is not None else []

    async def run_live(self, resume_text: str, job_description_text: str):
        # This async generator needs to be adapted for a direct API call.
        # For a backend, we just want the final result, not streaming messages.
        candidate_agent_found = next((agent for agent in self.sub_agents if agent.name == "candidate_agent"), None)
        if not candidate_agent_found:
            raise ValueError("candidate_agent not found.")

        analysis_tool_instance = next((tool for tool in candidate_agent_found.tools if tool.name == "analyze_resume_job_description"), None)
        if not analysis_tool_instance:
            raise ValueError("analyze_resume_job_description tool not found for candidate_agent.")

        analysis_result = analysis_tool_instance.func(resume_text, job_description_text)
        return analysis_result

# Re-define Tool and Agent instances for FastAPI context
analysis_tool = Tool(
    func=analyze_resume_job_description_full,
    name="analyze_resume_job_description",
    description="Analyzes a candidate's resume against a job description to identify skills and gaps using an LLM."
)

candidate_agent = Agent(
    name="candidate_agent",
    instruction="I manage candidate profiles and analyze resumes against job descriptions.",
    tools=[analysis_tool]
)

root_agent = CoordinatorAgent(
    name="root_agent",
    instruction="I orchestrate the resume and job description analysis process.",
    sub_agents=[candidate_agent]
)

app = FastAPI()

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description_text: str

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        # Call the synchronous analysis function directly or adapt CoordinatorAgent's run_live
        # For this backend, we'll call the core function that returns a dict
        analysis_result = root_agent.sub_agents[0].tools[0].func(request.resume_text, request.job_description_text)
        if analysis_result.get('analysis_status') == 'failure':
            raise HTTPException(status_code=500, detail=analysis_result)
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": str(e)})

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Resume Analyzer Backend is running"}

requirements.txt (for backend):

fastapi
uvicorn
python-dotenv
google-generativeai==0.8.5 # Ensure correct version based on previous steps
requests
beautifulsoup4
pypdf2
nest-asyncio
Dockerfile (for backend):

# Use the official Python image as a base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY backend_main.py .

# Expose the port that the application listens on
EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "backend_main:app", "--host", "0.0.0.0", "--port", "8000"]
Deployment Steps (Backend):

Save Files: Create backend_main.py, requirements.txt, and Dockerfile in a new directory (e.g., backend_service).
Authenticate GCP: Ensure you are authenticated to Google Cloud. If using Colab, you'd typically do this via gcloud auth login in your local terminal or configure Colab for Cloud SDK if deploying from Colab itself (though local deployment is usually preferred for Cloud Run).
Set Project: gcloud config set project YOUR_GCP_PROJECT_ID
Build and Push Docker Image:
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/ai-resume-analyzer-backend ./backend_service
Deploy to Cloud Run:
gcloud run deploy ai-resume-analyzer-backend \
  --image gcr.io/YOUR_GCP_PROJECT_ID/ai-resume-analyzer-backend \
  --platform managed \
  --region YOUR_GCP_REGION \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=YOUR_GEMINI_API_KEY \
  --max-instances 1 \
  --cpu 1 \
  --memory 2Gi \
  --timeout 300
Replace YOUR_GCP_PROJECT_ID, YOUR_GCP_REGION, and YOUR_GEMINI_API_KEY.
--allow-unauthenticated makes the service publicly accessible. Remove for private services.
--max-instances, --cpu, --memory, --timeout are important for performance and cost. Adjust as needed. LLM calls can be slow, so timeout should be generous.
Make note of the Service URL provided after deployment; this will be your BACKEND_API_URL.
2. Frontend Service Deployment (Streamlit Application)
The Streamlit frontend also needs to be containerized and deployed.

frontend_app.py (Same content as the last Streamlit code cell output):

# Save this content as frontend_app.py
import streamlit as st
import os
import io
import PyPDF2
import requests
from bs4 import BeautifulSoup
import json

# --- Helper function for URL extraction (copied from backend) ---
def extract_text_from_url(url: str) -> str:
    """Extracts text content from a given URL, typically for a job description."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing URL {url}: {e}")
        return ""

# --- Streamlit UI and Workflow Integration ---

st.set_page_config(
    page_title="AI-Powered Resume and Job Description Analyzer (Frontend)",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> 🔍 AI Job Search Assistant (Frontend) </h1> <p style='text-align:center; font-size:18px;'> Discover tailored job recommendations powered by Agentic AI. </p>", unsafe_allow_html=True)
st.sidebar.header("User Inputs")

# Hypothetical backend API endpoint
# In a real deployment, this URL would point to your deployed Google Cloud Run service.
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/analyze") # Placeholder URL for local testing or environment variable

st.sidebar.markdown(f"*(Backend API URL: `{BACKEND_API_URL}`)*")
st.sidebar.markdown("---")


job_url_input = st.sidebar.text_input(
    "Job Description URL",
    value="https://example.com/job_description",
    help="Enter the URL of the job description webpage."
)

uploaded_resume_file = st.sidebar.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"],
    help="Upload your resume in PDF format."
)

is_valid_job_url = False
if job_url_input:
    if job_url_input.startswith("http://") or job_url_input.startswith("https://"):
        is_valid_job_url = True
    else:
        st.sidebar.error("Please enter a valid URL (starting with http:// or https://).")

is_resume_uploaded = False
if uploaded_resume_file is not None:
    is_resume_uploaded = True

if st.sidebar.button("Run Analysis", disabled=(not is_valid_job_url or not is_resume_uploaded)):
    if is_valid_job_url and is_resume_uploaded:
        with st.spinner("Processing resume and fetching job description..."):
            resume_text = ""
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_resume_file.getvalue()))
                resume_text = "".join([page.extract_text() for page in pdf_reader.pages])
                st.success("Resume extracted successfully.")
            except Exception as e:
                st.error(f"Error reading resume PDF: {e}")
                resume_text = ""

            job_description_text = ""
            try:
                job_description_text = extract_text_from_url(job_url_input)
                if job_description_text:
                    st.success("Job description fetched successfully.")
                else:
                    st.error("Failed to fetch job description. Please check the URL.")
            except Exception as e:
                st.error(f"Error fetching job description from URL: {e}")
                job_description_text = ""

        if resume_text and job_description_text:
            st.subheader("AI Analysis Report")
            progress_bar = st.progress(0)
            status_text = st.empty()
            report_container = st.empty()

            status_text.text("Sending data to backend for analysis...")
            progress_bar.progress(25)

            try:
                payload = {
                    "resume_text": resume_text,
                    "job_description_text": job_description_text
                }
                headers = {"Content-Type": "application/json"}

                # Make a POST request to the backend API
                response = requests.post(BACKEND_API_URL, json=payload, headers=headers, timeout=120) # Increased timeout for LLM calls
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                analysis_result = response.json()

                if analysis_result.get('analysis_status') == 'success':
                    status_text.text("✅ Analysis complete: Received report from backend.")
                    progress_bar.progress(100)

                    parsed_report = analysis_result.get('parsed_report', {})
                    report_html_parts = []

                    report_html_parts.append("<hr/>")
                    # Overall Fit Summary
                    overall_fit_summary = parsed_report.get('overall_fit_summary', 'N/A')
                    report_html_parts.append(f"<p><b>Overall Fit Summary:</b> {overall_fit_summary}</p>")
                    report_html_parts.append("<hr/>")

                    # Candidate Skills
                    candidate_skills = parsed_report.get('candidate_skills', [])
                    if candidate_skills:
                        report_html_parts.append("<h4>Candidate Skills:</h4><ul>")
                        for skill in candidate_skills:
                            report_html_parts.append(f"<li>{skill}</li>")
                        report_html_parts.append("</ul>")

                    # Required Job Skills
                    required_job_skills = parsed_report.get('required_job_skills', [])
                    if required_job_skills:
                        report_html_parts.append("<h4>Required Job Skills:</h4><ul>")
                        for skill in required_job_skills:
                            report_html_parts.append(f"<li>{skill}</li>")
                        report_html_parts.append("</ul>")

                    # Matched Skills
                    matched_skills = parsed_report.get('matched_skills', [])
                    if matched_skills:
                        report_html_parts.append("<h4>Matched Skills:</h4><ul>")
                        for skill in matched_skills:
                            report_html_parts.append(f"<li>{skill}</li>")
                        report_html_parts.append("</ul>")

                    # Missing Skills
                    missing_skills = parsed_report.get('missing_skills', [])
                    if missing_skills:
                        report_html_parts.append("<h4 style=\"color:red;\">Missing Skills (Gaps):</h4><ul>")
                        for skill in missing_skills:
                            report_html_parts.append(f"<li style=\"color:red;\">{skill}</li>")
                        report_html_parts.append("</ul>")

                    # Additional Skills
                    additional_skills = parsed_report.get('additional_skills', [])
                    if additional_skills:
                        report_html_parts.append("<h4>Additional Skills:</h4><ul>")
                        for skill in additional_skills:
                            report_html_parts.append(f"<li>{skill}</li>")
                        report_html_parts.append("</ul>")

                    report_container.markdown("\n".join(report_html_parts), unsafe_allow_html=True)

                else:
                    status_text.text(f"❌ Analysis failed: {analysis_result.get('message', 'Unknown error from backend.')}")
                    st.error(f"Backend analysis failed: {analysis_result.get('message', 'Unknown error.')}")
                    if 'raw_report' in analysis_result:
                        st.json(analysis_result['raw_report']) # Display raw report if available for debugging
            except requests.exceptions.ConnectionError as ce:
                st.error(f"❌ Connection Error: Could not connect to the backend API at {BACKEND_API_URL}. Please ensure the backend is running and accessible. Details: {ce}")
            except requests.exceptions.Timeout as te:
                st.error(f"❌ Request Timeout: The backend API at {BACKEND_API_URL} took too long to respond. Details: {te}")
            except requests.exceptions.RequestException as re:
                st.error(f"❌ Error during API call to backend: {re}")
                if hasattr(re, 'response') and re.response is not None:
                    try:
                        st.error(f"Backend Response: {re.response.json()}")
                    except json.JSONDecodeError:
                        st.error(f"Backend Response (raw): {re.response.text}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during backend communication: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

        else:
            st.error("Analysis cannot be performed due to missing resume text or job description text.")
    else:
        st.error("Please fix the input errors before running analysis.")
else:
    if not is_valid_job_url or not is_resume_uploaded:
        st.warning("Please provide a valid Job URL and upload your resume to proceed.")
    else:
        st.success("Job URL and Resume uploaded successfully. Ready for analysis!")

st.markdown("""
---
**Note for Colab Users:**
This Streamlit application is designed to run as a separate frontend, interacting with a *deployed* backend API.
It cannot be run interactively within this Colab notebook due to how Streamlit serves applications and Colab's environment limitations.
To test this frontend, you would typically:
1. Save this code as `frontend_app.py`.
2. Deploy your backend AI logic (e.g., as a FastAPI service on Google Cloud Run).
3. Set the `BACKEND_API_URL` environment variable (or hardcode it for testing) to your deployed backend's URL.
4. Run `streamlit run frontend_app.py` in your local environment or deploy it to a service like Streamlit Community Cloud or Google Cloud Run.
""", unsafe_allow_html=True)
.streamlit/config.toml (Optional, for Streamlit server configuration):

# Save this content as .streamlit/config.toml inside a .streamlit folder
[server]
port = 8080
enableCORS = false
headless = true
requirements.txt (for frontend):

streamlit==1.33.0
PyPDF2==3.0.1
requests==2.32.3
beautifulsoup4==4.12.3
Dockerfile (for frontend):

# Use the official Python image as a base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the Streamlit config folder
COPY .streamlit/ .streamlit/

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application file into the container
COPY frontend_app.py .

# Expose the port that the Streamlit app listens on
EXPOSE 8080

# Command to run the Streamlit application
CMD ["streamlit", "run", "frontend_app.py", "--server.port=8080", "--server.enableCORS=false", "--server.headless=true"]
Deployment Steps (Frontend):

Save Files: Create frontend_app.py, requirements.txt, and a .streamlit folder containing config.toml in a new directory (e.g., frontend_service).
Authenticate GCP: Ensure you are authenticated to Google Cloud.
Set Project: gcloud config set project YOUR_GCP_PROJECT_ID
Build and Push Docker Image:
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/ai-resume-analyzer-frontend ./frontend_service
Deploy to Cloud Run:
gcloud run deploy ai-resume-analyzer-frontend \
  --image gcr.io/YOUR_GCP_PROJECT_ID/ai-resume-analyzer-frontend \
  --platform managed \
  --region YOUR_GCP_REGION \
  --allow-unauthenticated \
  --set-env-vars BACKEND_API_URL=YOUR_DEPLOYED_BACKEND_SERVICE_URL \
  --max-instances 1 \
  --cpu 1 \
  --memory 1Gi \
  --timeout 300
Replace YOUR_GCP_PROJECT_ID, YOUR_GCP_REGION, and YOUR_DEPLOYED_BACKEND_SERVICE_URL (this is the URL you got from the backend deployment).
--allow-unauthenticated makes the service publicly accessible.
Streamlit apps can be memory-intensive; adjust --memory as needed.