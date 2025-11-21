### Problem Statement 
```
Job search today is painfully fragmented and inefficient.

Candidates:

•	Manually copy-paste job descriptions across multiple sites

•	Struggle to understand their skill gaps

•	Waste hours crafting tailored applications

•	Lack insights or guidance at each step

Recruiters and job platforms aren’t solving this - everything is static, keyword-based, and not intuitive.

The world needs an intelligent, proactive job assistant that understands natural language, searches jobs automatically, analyzes resumes, and guides career decisions.
```
### Why agents? 
```
A single LLM prompt isn’t enough.
```
#### Job search is a workflow, not a one-shot response. It requires:
```
•	Searching external job APIs

•	Parsing resumes

•	Extracting skills

•	Performing reasoning

•	Deciding which tool to call

•	Asking follow-up questions

•	Providing actionable next steps
```

#### Agents are perfect for this because they can:
```
•	Break complex tasks into sub-tasks

•	Call tools (job search API, resume parsers, ranking algorithms)

•	Iterate, reflect, and refine outputs

•	Deliver end-to-end automation, not just text

This turns the LLM from a chatbot → a smart orchestrator that gets things done.
```

### What I created 
```
I built a full Agentic AI Job Assistant, consisting of:
```

#### Frontend (User Interface)
```
•	Streamlit app

•	Resume upload

•	Real-time agentic reasoning displayed
```

#### Coordinator Agent (Brain)
```
•	OpenAI LLM (4o-mini)

•	Multi-step reasoning

•	Tool-calling enabled
```

#### Agent Tools
```
•	Resume Parser → Converts resume into structured JSON

•	Skill Gap Analyzer → Compares resume vs job requirements

•	Application Advisor → Summarizes fit & next actions
```

### Demo  
```
•	Solution is deployed to Streamlit Cloud: https://lexman29125.streamlit.app/

•	Attached images with description guide the workflow process and agents interactions
```

### The Build 
```
The solution has demonstrated the following key concepts:
1. Multi-agent system:

-	Agent powered by an LLM

-	Sequential agents

2. Tools, including:

-	custom tools

-	OpenAPI tools

3. Agent deployment:

-	Deployed to Streamlit Cloud

-	Deployable to:

-	Google Cloud Run

-	Hugging Face Spaces

-	AWS Lightsail

-	Render
```

#### Backend
##### Built a custom agent framework:
```
•	Defined Tool and Agent classes from scratch.

•	Built a hierarchical CoordinatorAgent → CandidateAgent architecture.

•	Implemented a custom async generator agent workflow (run_live) to stream step-by-step results.

•	Designed a modular system where each tool has:

-	a function

-	a name

-	a description
```

##### Integrated OpenAI’s 4.1-mini model for reasoning
```
•	Used openai Python SDK with OpenAI api_key

•	Built a JSON reasoning prompt to extract:

-	skills

-	missing skills

-	matched skills

-	candidate vs JD comparison

-	Added full error handling and safe parsing logic.

This transforms raw resume + job description into structured insights.
```

##### Implemented job description extraction
```
•	Requests

•	BeautifulSoup
```

##### Created custom scraper that:
```
•	Fetches job pages

•	Removes scripts/styles

•	Cleans text

•	Produces plain raw JD text for the LLM

This ensures the LLM sees clean job content instead of HTML noise.
```

##### Implemented PDF resume extraction
```
•	PyPDF2

•	io.BytesIO

The app extracts text from every PDF page and merges the results into a clean string for LLM analysis.
```

##### Streamlit App UI
```
Transformed the agent into an interactive web app:
•	Sidebar inputs

•	Resume upload

•	URL for job description

•	Progress bar

•	Step-by-step streaming output

•	HTML-styled analysis report

This gives the feeling of a live AI assistant.
```

##### Security / Configuration
```
•	.env loading via dotenv

•	Validated OpenAI API key

•	Stopped the app if key missing

•	Provided user-friendly Streamlit sidebar warnings
```

##### Async Execution Integrated with Streamlit
```
•	Used asyncio.run() inside Streamlit

•	Used nest_asyncio.apply() to patch the event loop

•	Wrapped the agent’s async workflow into a usable UI flow

This gives real-time progress updates during analysis.
```

#### Frontend
```
•	Code wrapper written for Streamlit
```

#### Agent Deployment
```
•	Works locally, and deployable to:

•	Streamlit Cloud: https://lexman29125.streamlit.app/

•	Google Cloud

•	Hugging Face Spaces

•	AWS Lightsail

•	Render
```

### If I had more time, this is what I'd do
```
•	Chat-style UI

•	Context Engineering, Sessions & Memory: Maintains memory of conversation history

•	Allows user feedback to fine-tune Resume to better position for the desired job

•	MCP & Google Search tools: Job Search Tool → Calls external job APIs (e.g., FindSGJobs)

•	Downloadable report

•	Agent evaluation with Observability: Logging, Tracing, Metrics
```
