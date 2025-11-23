export interface AnalysisResult {
  overallFitSummary: string;
  candidateSkills: string[];
  requiredJobSkills: string[];
  gapAnalysis: string[];
  matchScore: number; // 0 to 100
}

// CONCEPT: Observability & Tracing
// structured logging for agent activities
export interface AgentTrace {
  id: string;
  agentName: string;
  action: string;
  status: 'started' | 'completed' | 'failed';
  timestamp: number;
  duration?: number;
}

export interface LogEntry {
  id: string;
  message: string;
  type: 'info' | 'success' | 'agent' | 'error';
  timestamp: number;
  agentName?: string; // To identify which agent generated the log
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export type ViewMode = 'analysis' | 'cv_preview';

// CONCEPT: Session State Management
export interface AppState {
  jobUrl: string;
  resumeFile: File | null;
  isLoading: boolean;
  logs: LogEntry[];
  result: AnalysisResult | null;
  error: string | null;
  chatMessages: ChatMessage[];
  viewMode: ViewMode;
  generatedCV: string | null;
}
