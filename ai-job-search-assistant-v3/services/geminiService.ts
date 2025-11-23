import { GoogleGenAI, Chat } from "@google/genai";
import { AnalysisResult, AgentTrace, LogEntry } from "../types";

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- CONCEPT: Observability & Metrics ---
// A simple service to track agent performance and execution flow.
class TelemetryService {
  private traces: AgentTrace[] = [];
  private logCallback: (log: LogEntry) => void;

  constructor(logCallback: (log: LogEntry) => void) {
    this.logCallback = logCallback;
  }

  startSpan(agentName: string, action: string): string {
    const id = Math.random().toString(36).substring(7);
    const timestamp = Date.now();
    this.traces.push({ id, agentName, action, status: 'started', timestamp });
    
    // Emit to UI log
    this.logCallback({
      id,
      message: `${action}...`,
      type: 'agent',
      timestamp,
      agentName
    });
    
    return id;
  }

  endSpan(spanId: string, status: 'completed' | 'failed', message?: string) {
    const span = this.traces.find(t => t.id === spanId);
    if (span) {
      span.status = status;
      span.duration = Date.now() - span.timestamp;
      
      this.logCallback({
        id: Math.random().toString(36).substring(7),
        message: message || `${span.action} ${status} (${span.duration}ms)`,
        type: status === 'completed' ? 'success' : 'error',
        timestamp: Date.now(),
        agentName: span.agentName
      });
    }
  }
}

// --- CONCEPT: Memory & State Management ---
// Implements a "Memory Bank" pattern.
// Separates "Long Term Memory" (Analysis Facts) from "Short Term Memory" (Chat Context).
class MemoryBank {
  // Long-term memory: Structured facts derived from heavy processing
  public analysisContext: AnalysisResult | null = null;
  public jobDescription: string | null = null;
  public resumeText: string | null = null;
  
  // Short-term memory: conversational history is handled by the Chat object, 
  // but we can store critical user preferences here.
  public userPreferences: Record<string, any> = {};

  // CONCEPT: Context Compaction
  // When feeding the LLM, we don't send the entire raw PDF text every time.
  // We send this compacted summary.
  getCompactedContext(): string {
    if (!this.analysisContext) return "";
    return `
      Match Score: ${this.analysisContext.matchScore}/100.
      Top Skills: ${this.analysisContext.candidateSkills.slice(0, 5).join(', ')}.
      Critical Gaps: ${this.analysisContext.gapAnalysis.join(', ')}.
    `;
  }
}

// --- CONCEPT: Agent Abstraction ---
// Base class for all agents to ensure consistent interface and observability.
abstract class BaseAgent {
  protected name: string;
  protected model: string = 'gemini-2.5-flash';
  protected telemetry: TelemetryService;

  constructor(name: string, telemetry: TelemetryService) {
    this.name = name;
    this.telemetry = telemetry;
  }

  abstract run(...args: any[]): Promise<any>;
}

// --- Agent 1: Job Architect (Tool-Enabled Agent) ---
// CONCEPT: Tools (Built-in) -> Uses Google Search to find/parse JD.
class JobArchitectAgent extends BaseAgent {
  async run(jobUrl: string): Promise<string> {
    const spanId = this.telemetry.startSpan(this.name, `Researching Job at ${new URL(jobUrl).hostname}`);
    
    try {
      // CONCEPT: Tool Use
      // We enable googleSearch. The model decides how to use it to get the JD content.
      const response = await ai.models.generateContent({
        model: this.model,
        contents: {
          parts: [{ text: `Find the full job description text for this URL: ${jobUrl}. Return a comprehensive summary of the responsibilities and requirements.` }]
        },
        config: {
          tools: [{ googleSearch: {} }],
        }
      });

      const text = response.text || "";
      if (!text) throw new Error("Could not fetch job description");
      
      this.telemetry.endSpan(spanId, 'completed');
      return text;
    } catch (e) {
      this.telemetry.endSpan(spanId, 'failed', "Failed to fetch job description");
      throw e;
    }
  }
}

// --- Agent 2: Resume Analyst (Multimodal Agent) ---
// CONCEPT: Multimodal Input -> Processes raw PDF bytes.
class ResumeAnalystAgent extends BaseAgent {
  async run(file: File): Promise<{ inlineData: { data: string; mimeType: string } }> {
    const spanId = this.telemetry.startSpan(this.name, "Processing PDF Document");
    
    // Simulating "Processing" time for realism in UI
    await new Promise(r => setTimeout(r, 500));

    const fileData = await this.fileToGenerativePart(file);
    
    this.telemetry.endSpan(spanId, 'completed', "Resume extracted");
    return fileData;
  }

  private async fileToGenerativePart(file: File): Promise<{ inlineData: { data: string; mimeType: string } }> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Data = reader.result as string;
        resolve({
          inlineData: {
            data: base64Data.split(',')[1],
            mimeType: file.type,
          },
        });
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
}

// --- Agent 3: Gap Analyst (Logic Agent) ---
// CONCEPT: Sequential Agent -> Runs only after data is ready.
// CONCEPT: Structured Output -> Enforces JSON schema via Prompt Engineering (or Config).
class GapAnalystAgent extends BaseAgent {
  async run(resumePart: any, jobContext: string): Promise<AnalysisResult> {
    const spanId = this.telemetry.startSpan(this.name, "Performing Gap Analysis");

    const prompt = `
      You are an expert Technical Recruiter.
      
      Job Context: ${jobContext}
      
      Task: Analyze the provided Resume (PDF) against the Job Context above.
      
      Output Requirements:
      Return a single valid JSON object. No markdown formatting.
      Structure:
      {
        "overallFitSummary": "string",
        "candidateSkills": ["string"],
        "requiredJobSkills": ["string"],
        "gapAnalysis": ["string"],
        "matchScore": number (0-100)
      }
    `;

    const response = await ai.models.generateContent({
      model: this.model,
      contents: {
        parts: [
          resumePart,
          { text: prompt }
        ]
      }
    });

    let jsonString = response.text?.trim() || "{}";
    // Sanitize
    if (jsonString.startsWith('```')) {
      jsonString = jsonString.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '');
    }
    const startIndex = jsonString.indexOf('{');
    const endIndex = jsonString.lastIndexOf('}');
    if (startIndex !== -1 && endIndex !== -1) {
      jsonString = jsonString.substring(startIndex, endIndex + 1);
    }

    try {
      const result = JSON.parse(jsonString);
      this.telemetry.endSpan(spanId, 'completed', `Analysis complete. Score: ${result.matchScore}`);
      return result;
    } catch (e) {
      this.telemetry.endSpan(spanId, 'failed');
      throw e;
    }
  }
}

// --- Agent 4: Career Coach (Chat Agent) ---
// CONCEPT: Interactive Agent with Personality & Memory
export const createChatSession = (memory: MemoryBank): Chat => {
  const systemInstruction = `
    You are a dedicated Career Coach.
    
    // CONCEPT: Context Injection / Compaction
    Current Analysis Context:
    ${memory.getCompactedContext()}
    
    Goal: Help the candidate land this job.
    
    Capabilities:
    1. Answer questions about the gaps.
    2. "Draft CV": If asked, generate a full ATS-compliant Markdown CV.
       - Use standard headers (# Name, ## Experience).
       - No conversational filler when drafting.
    
    // CONCEPT: Loop / Iteration
    If the user asks for changes to the CV, regenerate the WHOLE CV with improvements.
  `;

  return ai.chats.create({
    model: 'gemini-2.5-flash',
    config: { systemInstruction }
  });
};

// --- Main Service Export (The Orchestrator) ---
// CONCEPT: Orchestrator Pattern -> Manages the lifecycle of multiple agents.
export const orchestrateAnalysis = async (
  resumeFile: File,
  jobUrl: string,
  logCallback: (log: LogEntry) => void
): Promise<{ result: AnalysisResult; memory: MemoryBank }> => {
  
  const telemetry = new TelemetryService(logCallback);
  const memory = new MemoryBank();

  // Instantiate Agents
  const jobAgent = new JobArchitectAgent("Job_Architect_Agent", telemetry);
  const resumeAgent = new ResumeAnalystAgent("Resume_Parser_Agent", telemetry);
  const gapAgent = new GapAnalystAgent("Gap_Analysis_Agent", telemetry);

  try {
    // CONCEPT: Parallel Execution
    // We launch the Job Research and Resume Processing simultaneously to save time.
    logCallback({ id: 'orch-1', message: 'Orchestrator starting parallel agents...', type: 'info', timestamp: Date.now() });
    
    const [jobContext, resumePart] = await Promise.all([
      jobAgent.run(jobUrl),
      resumeAgent.run(resumeFile)
    ]);

    // Store in Memory
    memory.jobDescription = jobContext;
    
    // CONCEPT: Sequential Execution
    // Gap Analysis depends on the output of the previous parallel agents.
    const result = await gapAgent.run(resumePart, jobContext);
    memory.analysisContext = result;

    return { result, memory };

  } catch (error) {
    console.error("Orchestration failed", error);
    throw error;
  }
};

export const sendChatMessage = async (chat: Chat, message: string): Promise<string> => {
  const response = await chat.sendMessage({ message });
  return response.text || "I'm sorry, I couldn't generate a response.";
};
