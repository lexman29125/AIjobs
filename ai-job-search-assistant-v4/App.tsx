import React, { useState, useCallback, useRef } from 'react';
import { Chat } from '@google/genai';
import Sidebar from './components/Sidebar';
import MainContent from './components/MainContent';
import { LogEntry, AnalysisResult, ChatMessage, ViewMode } from './types';
import { orchestrateAnalysis, createChatSession, sendChatMessage } from './services/geminiService';

const App: React.FC = () => {
  const [jobUrl, setJobUrl] = useState('');
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  
  // Chat & CV State
  const chatSessionRef = useRef<Chat | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('analysis');
  const [generatedCV, setGeneratedCV] = useState<string | null>(null);

  // Callback for TelemetryService to push logs to UI
  const handleLog = useCallback((log: LogEntry) => {
    setLogs(prev => [...prev, log]);
  }, []);

  const handleRunAnalysis = async () => {
    if (!jobUrl || !resumeFile) return;

    setIsLoading(true);
    setLogs([]); 
    setResult(null);
    setChatMessages([]);
    setGeneratedCV(null);
    setViewMode('analysis');
    chatSessionRef.current = null;

    try {
      const { result, memory } = await orchestrateAnalysis(
        resumeFile, 
        jobUrl, 
        handleLog
      );
      
      setResult(result);
      chatSessionRef.current = createChatSession(memory);

    } catch (error) {
      console.error(error);
      handleLog({
        id: 'err', 
        message: "An error occurred during analysis.", 
        type: 'info', 
        timestamp: Date.now()
      });
      if (error instanceof Error) {
         handleLog({
           id: 'err-detail', 
           message: `Error details: ${error.message}`, 
           type: 'info', 
           timestamp: Date.now()
         });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendChatMessage = async (text: string) => {
    if (!chatSessionRef.current) return;

    setChatMessages(prev => [...prev, { role: 'user', text }]);
    setIsChatLoading(true);

    try {
      const response = await sendChatMessage(chatSessionRef.current, text);
      const isCV = response.trim().startsWith('# ') || (response.includes('## Experience') && response.length > 300);

      if (viewMode === 'cv_preview' && isCV) {
        setGeneratedCV(response);
        setChatMessages(prev => [...prev, { role: 'model', text: "✅ I've updated the CV draft based on your feedback." }]);
      } else {
        setChatMessages(prev => [...prev, { role: 'model', text: response }]);
      }
      
    } catch (error) {
      console.error("Chat error", error);
      setChatMessages(prev => [...prev, { role: 'model', text: "Sorry, I encountered an error processing your message." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleDraftCV = async () => {
    if (!chatSessionRef.current) return;

    const prompt = "Based on our discussion and the initial analysis, please draft the full proposed CV now in Markdown format. Remember: Output ONLY the Markdown content.";
    
    setChatMessages(prev => [...prev, { role: 'user', text: "Draft Proposed CV for the job" }]);
    setIsChatLoading(true);

    try {
      const cvContent = await sendChatMessage(chatSessionRef.current, prompt);
      
      setChatMessages(prev => [...prev, { role: 'model', text: "I have drafted the tailored CV. You can now review it on the split screen." }]);
      
      setTimeout(() => {
        setGeneratedCV(cvContent);
        setViewMode('cv_preview');
      }, 500);
      
    } catch (error) {
      console.error("CV Generation error", error);
      setChatMessages(prev => [...prev, { role: 'model', text: "Sorry, I failed to generate the CV. Please try again." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 font-sans overflow-hidden">
      <Sidebar 
        jobUrl={jobUrl}
        setJobUrl={setJobUrl}
        resumeFile={resumeFile}
        setResumeFile={setResumeFile}
        onRunAnalysis={handleRunAnalysis}
        isLoading={isLoading}
      />
      <MainContent 
        logs={logs}
        result={result}
        isLoading={isLoading}
        chatMessages={chatMessages}
        onSendChatMessage={handleSendChatMessage}
        isChatLoading={isChatLoading}
        onDraftCV={handleDraftCV}
        viewMode={viewMode}
        generatedCV={generatedCV}
        onBackToAnalysis={() => setViewMode('analysis')}
      />
    </div>
  );
};

export default App;