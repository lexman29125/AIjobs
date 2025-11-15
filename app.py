import streamlit as st
import os
import io
import PyPDF2
import asyncio
import nest_asyncio
import random
import re
import json
import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# Load .env file
load_dotenv()

# Ensure nest_asyncio is applied if not already done in the session
nest_asyncio.apply()

# Helper function to parse salary from string (might not be used in current workflow but kept for consistency)
def parse_salary(salary_str: str) -> int:
    if not salary_str: return 0
    # Remove non-numeric characters except comma, then remove comma, then convert to int
    numeric_str = re.sub(r'[^\\d,]', '', salary_str)
    numeric_str = numeric_str.replace(',', '')
    try:
        return int(numeric_str)
    except ValueError:
        return 0

# Tool and Agent class definitions
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

# Helper function to extract text from URL
def extract_text_from_url(url: str) -> str:
    """Extracts text content from a given URL, typically for a job description."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Modified: Increased timeout from 10 to 30 seconds
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
        st.error(f"Error fetching URL {url}: {e}") # Use st.error for Streamlit
        return ""
    except Exception as e:
        st.error(f"Error processing URL {url}: {e}") # Use st.error for Streamlit
        return ""

# OpenAI Model setup
openai_client = None
llm_model_name = "gpt-4o-mini" # Default to a commonly available OpenAI model

# Configure the OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# In Streamlit, handle API key input/error more gracefully
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    st.sidebar.error("OPENAI_API_KEY environment variable not set or is a placeholder.")
    st.sidebar.warning("Please provide your OpenAI API Key to continue. (Set as environment variable or in st.secrets)")
    st.stop() # Stop the app if API key is missing

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    # st.sidebar.success(f"OpenAI client initialized with model: {llm_model_name}") # Suppress this print in the final UI
except Exception as e:
    st.sidebar.error(f"Error initializing OpenAI client: {e}. Please check your API key and network connection.")
    st.stop()

def analyze_skills_and_gaps(resume_text: str, job_description_text: str) -> str:
    """Analyzes a candidate's resume against a job description using the LLM to identify skills and gaps."""
    system_prompt = """You are an expert HR analyst. Your task is to compare a candidate's resume with a job description. \
    Provide your output as a JSON object ONLY. Do not include any other text or explanation outside the JSON."""

    user_prompt = f"""Here is the candidate's Resume:
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
        if openai_client is None:
            return "Error: OpenAI client not initialized."

        response = openai_client.chat.completions.create(
            model=llm_model_name,
            response_format={ "type": "json_object" }, # Specify JSON output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during LLM analysis: {e}"

def analyze_resume_job_description_full(resume_text: str, job_description_text: str) -> dict:
    """Performs a full resume and job description analysis using the LLM.
    This function replaces the placeholder and calls analyze_skills_and_gaps.
    """
    # Use st.info for Streamlit progress updates
    st.info(f"Initiating LLM-based analysis for resume (length: {len(resume_text)}) and job description (length: {len(job_description_text)}).")
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

# Re-define Tool instances
analysis_tool = Tool(
    func=analyze_resume_job_description_full,
    name="analyze_resume_job_description",
    description="Analyzes a candidate's resume against a job description to identify skills and gaps using an LLM."
)

# Re-define Agent instances
candidate_agent = Agent(
    name="candidate_agent",
    instruction="I manage candidate profiles and analyze resumes against job descriptions.",
    tools=[analysis_tool]
)

# Redefine CoordinatorAgent to reflect new workflow
class CoordinatorAgent(Agent):
    def __init__(self, name: str, instruction: str, tools: list = None, sub_agents: list = None):
        super().__init__(name, instruction, tools if tools is not None else [])
        self.sub_agents = sub_agents if sub_agents is not None else []

    async def run_live(self, resume_text: str, job_description_text: str):
        yield f"🚀 CoordinatorAgent '{self.name}' initiating resume and job description analysis..."

        candidate_agent_found = next((agent for agent in self.sub_agents if agent.name == "candidate_agent"), None)
        if not candidate_agent_found:
            yield "❌ Error: candidate_agent not found."
            return

        analysis_tool_instance = next((tool for tool in candidate_agent_found.tools if tool.name == "analyze_resume_job_description"), None)
        if not analysis_tool_instance:
            yield "❌ Error: analyze_resume_job_description tool not found for candidate_agent."
            return

        yield f"⚙️ Delegating analysis to {candidate_agent_found.name} using {analysis_tool_instance.name} tool..."
        try:
            analysis_result = analysis_tool_instance.func(resume_text, job_description_text)
            if analysis_result.get('analysis_status') == 'success':
                yield f"✅ Analysis complete: {analysis_result.get('message', 'No message provided.')}"
                yield "<h2>Analysis Report</h2>" # Moved here to display after initial messages

                parsed_report = analysis_result.get('parsed_report', {})

                report_html_parts = []

                # Overall Fit Summary
                overall_fit_summary = parsed_report.get('overall_fit_summary', 'N/A')
                report_html_parts.append(f"<p><b>Overall Fit Summary:</b> {overall_fit_summary}</p>")

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

                yield "\n".join(report_html_parts) # Yield the complete HTML string

            else:
                yield f"❌ Analysis failed: {analysis_result.get('message', 'Unknown error.')}"
                if 'raw_report' in analysis_result:
                    yield f"Raw LLM output: {analysis_result['raw_report']}"
        except Exception as e:
            yield f"❌ Error during analysis: {e}"
            return

# Re-instantiate the CoordinatorAgent with the new class definition and updated sub-agents
root_agent = CoordinatorAgent(
    name="root_agent",
    instruction="I orchestrate the resume and job description analysis process.",
    sub_agents=[candidate_agent]
)


# --- Streamlit UI and Workflow Integration ---

st.set_page_config(
    page_title="AI-Powered Resume and Job Description Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> 🔍 AI Job Search Assistant </h1> <p style='text-align:center; font-size:18px;'> Discover tailored job recommendations powered by Agentic AI. </p>", unsafe_allow_html=True)
st.sidebar.header("User Inputs")

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
            # st.subheader("Analysis Report") # Removed this line
            progress_bar = st.progress(0)
            status_text = st.empty()
            report_container = st.empty()

            async def run_analysis_workflow_streamlit(res_text: str, jd_text: str):
                output_lines = []
                total_steps = 7
                current_step = 0
                async for step_output in root_agent.run_live(res_text, jd_text):
                    output_lines.append(step_output)
                    status_text.text(step_output)
                    current_step += 1
                    progress_bar.progress(min(current_step / total_steps, 1.0))
                return output_lines

            st.write("Starting AI analysis...")
            full_report_lines = []
            # Use asyncio.run to execute the async generator
            for step_output in asyncio.run(run_analysis_workflow_streamlit(resume_text, job_description_text)):
                 full_report_lines.append(step_output)

            progress_bar.empty()
            status_text.empty()
            report_container.markdown("\n".join(full_report_lines), unsafe_allow_html=True)

        else:
            st.error("Analysis cannot be performed due to missing resume text or job description text.")
    else:
        st.error("Please fix the input errors before running analysis.")
else:
    if not is_valid_job_url or not is_resume_uploaded:
        st.warning("Please provide a valid Job URL and upload your resume to proceed.")
    else:
        st.success("Job URL and Resume uploaded successfully. Ready for analysis!")