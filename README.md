### Problem Statement 
```
In today's competitive job market, the "spray and pray" method of applying with a generic resume rarely works. Candidates struggle to tailor their CVs for Applicant Tracking Systems (ATS) because deciphering job descriptions and objectively analyzing one's own skills gap is time-consuming and prone to bias. Job seekers need a personalized career strategist, but human consultants are expensive and slow.
```
### Why agents? 
```
A single LLM prompt cannot effectively handle the complexity of this workflow. Reading a raw PDF, browsing the live web for a job description, performing a strict gap analysis, and then switching context to become an empathetic career coach requires distinct cognitive states.
Agents are essential here because:
•	Specialization: One agent parses complex multimodal data (PDFs), while another handles external tool execution (Google Search), preventing hallucination.
•	Latency: Parallel execution allows us to process heavy inputs simultaneously.
•	State: A dedicated memory bank is required to carry factual analysis into a creative writing context without losing details.
```
### What I created 
```
I built a Client-Side Agentic Application that acts as an autonomous career consultant. It doesn't just "chat"; it performs research, analyzes data, and executes work products (CV drafts).
•	Landing Page allowing user interaction by entering desired job URL and attach CV
•	Run Analysis to provide Executive Summary with profile match percentage against job requirements 
•	Summary of candidate's strengths vs job requirements
•	Critical gaps listing "deal-breakers"
•	Strategic Discussion function utilises a chatbot UI, allowing user to interact and feedback about the analysis and gaps
•	Draft Proposed CV uses the inputs from the chatbot user interaction to fine-tune the analysis, and produce a ATS standard CV that can be easily copied or downloaded for job application 
```
### Features included
```
This project implements a robust Multi-Agent System with the following key concepts embedded directly in the code:
1.	Orchestrator Pattern: A central controller (orchestrateAnalysis) manages the lifecycle of the operation, dispatching tasks and handling errors.
2.	Parallel Agents: The JobArchitectAgent (Web Search) and ResumeAnalystAgent (PDF Processing) run simultaneously using Promise.all to minimize user wait time.
3.	Sequential Agents: The GapAnalystAgent strictly waits for the outputs of the parallel agents, ensuring it has complete context before performing logic.
4.	Memory Bank: A dedicated class (MemoryBank) that separates Short-term memory (conversation history) from Long-term memory (hard analysis facts), implementing Context Compaction to keep the LLM focused.
5.	Observability: A custom TelemetryService that traces unique span IDs, execution times, and agent statuses, rendering them directly to the user UI.
6.	Tools: Explicit definition of the googleSearch tool attached specifically to the Job Architect agent, while other agents are restricted to internal reasoning.
7. Agent deployment to Google CloudRun
```
### The Build 
```
•	Core Logic: TypeScript
•	Framework: React 19
•	AI Engine: Google Gemini API (@google/genai SDK)
•	Model: gemini-2.5-flash (Optimized for speed and tool use)
•	Styling: Tailwind CSS
•	Icons: Lucide React
•	Architecture: Client-side Multi-Agent Orchestration with Session Memory

Frontend (User Interface)
•	React 19 & Tailwind CSS: A split-screen interface designed for productivity.
•	Live Observability: A real-time log window that visualizes agent thoughts, state transitions, and tool usage, giving the user transparency into the AI's reasoning.
•	Interactive Workspace: Includes a "Paper View" for CV drafting that mimics real-world document editing, alongside a chat interface for iterative refinement.

Backend (Agent Logic)
•	Browser-Based Orchestration: Instead of a heavy server, the application runs a sophisticated geminiService locally. It utilizes the Google GenAI SDK to manage a fleet of specialized agents.
•	Context Compaction: The system efficiently manages token usage by summarizing high-dimensional data (entire resumes/job posts) into compacted context blocks for the conversational agents.

Agent Tools
•	Google Search Tool: Enabled for the Job Architect Agent to fetch, scrape, and parse live job descriptions from LinkedIn or other job boards (grounding the AI in reality).
•	Multimodal Processing: The Resume Analyst Agent processes raw PDF bytes directly.
•	Structured Output Parsing: The Gap Analyst Agent is constrained to produce strict JSON for programmatic UI updates.
```

#### Agent Deployment
```
•	Google CloudRun
```

### Demo  
```
•	Solution is deployed to Google CloudRun: https://ai-job-search-assistant-v3-771689288896.us-west1.run.app/
```
```
•	Attached images with description guide the workflow process and agents interactions
```
•	Landing Page allowing user interaction by entering desired job URL and attach CV

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo1.png)

•	Run Analysis with observabilty and tracing of agents reasoning

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo2.png)

•	Run Analysis to provide Executive Summary with profile match percentage against job requirements. Provide summary of candidate's strengths vs job requirements and critical gaps listing "deal-breakers"

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo3.png)

•	Strategic Discussion function utilises a chatbot UI, allowing user to interact and feedback about the analysis and gaps

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo4.png)

•	Draft Proposed CV uses the inputs from the chatbot user interaction to fine-tune the analysis, and produce a ATS standard CV that can be easily copied or downloaded for job application 

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo5.png)

•	Downloaded Final CV compliant with ATS standard in WORD format

![Logo](https://github.com/lexman29125/AIjobs/raw/main/images/Demo6.png)

### If I had more time, this is what I'd do
```
•	Context Engineering, Sessions & Memory: Maintains memory of conversation history by creating a user profile via login. This requires a secured login and authentication mechanism to protect user Personal Identifiable Information (PII)

•	MCP & extend Google Search tools: Job Search Agent → Calls external job APIs (e.g., FindSGJobs, Linkedin etc)

•	Agent evaluation with Observability: Logging, Tracing, Metrics
```