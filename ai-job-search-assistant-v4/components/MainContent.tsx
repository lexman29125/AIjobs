import React, { useRef, useEffect } from 'react';
import { Search, CheckCircle, Activity, AlertCircle, Briefcase, UserCheck, ArrowRight, MessageSquare, Send, FileText, ArrowLeft, Copy, Download, Bot, User, Home, Users, BarChart2, Terminal } from 'lucide-react';
import { AnalysisResult, LogEntry, ChatMessage, ViewMode } from '../types';

interface ChatInterfaceProps {
  chatMessages: ChatMessage[];
  isChatLoading: boolean;
  inputMessage: string;
  setInputMessage: (val: string) => void;
  onSendMessage: (e: React.FormEvent) => void;
  viewMode: ViewMode;
  chatEndRef: React.RefObject<HTMLDivElement | null>;
  heightClass?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  chatMessages, 
  isChatLoading, 
  inputMessage, 
  setInputMessage, 
  onSendMessage, 
  viewMode, 
  chatEndRef, 
  heightClass = "h-[500px]" 
}) => {
  return (
    <div className={`bg-white border border-slate-200 rounded-xl overflow-hidden flex flex-col ${heightClass} shadow-sm`}>
      <div className="bg-slate-100 px-4 py-3 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot size={18} className="text-slate-700" />
          <span className="font-semibold text-slate-700 text-sm">AI Career Coach</span>
        </div>
        <span className="text-xs text-slate-400 font-medium">Powered by Gemini 2.5</span>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white">
          {chatMessages.length === 0 && (
             <div className="flex flex-col items-center justify-center h-full text-center space-y-3 opacity-50">
                <MessageSquare size={40} className="text-slate-300" />
                <p className="text-sm text-slate-500">Start a conversation to refine your strategy.</p>
             </div>
          )}
          {chatMessages.map((msg, idx) => (
            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'model' && (
                  <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center shrink-0 border border-slate-200">
                    <Bot size={16} className="text-slate-600" />
                  </div>
                )}
                <div className={`
                  px-4 py-2.5 rounded-2xl max-w-[85%] text-sm shadow-sm leading-relaxed
                  ${msg.role === 'user' 
                      ? 'bg-slate-800 text-white rounded-tr-none' 
                      : 'bg-slate-50 border border-slate-200 text-slate-800 rounded-tl-none'
                  }
                `}>
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
            </div>
          ))}
          {isChatLoading && (
            <div className="flex justify-start gap-3">
                 <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center shrink-0 border border-slate-200">
                    <Bot size={16} className="text-slate-600" />
                  </div>
                <div className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                  <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                  </div>
                </div>
            </div>
          )}
          <div ref={chatEndRef} />
      </div>
      <div className="bg-white border-t border-slate-200 p-3">
          <form onSubmit={onSendMessage} className="flex gap-2">
            <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder={viewMode === 'cv_preview' ? "Tell the AI to adjust the CV..." : "Ask for advice..."}
                className="flex-1 px-4 py-2.5 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent text-sm bg-slate-50 focus:bg-white transition-colors"
                disabled={isChatLoading}
            />
            <button 
                type="submit"
                disabled={!inputMessage.trim() || isChatLoading}
                className="bg-slate-800 text-white p-2.5 rounded-lg hover:bg-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm active:scale-95"
            >
                <Send size={18} />
            </button>
          </form>
      </div>
    </div>
  );
};

interface MainContentProps {
  logs: LogEntry[];
  result: AnalysisResult | null;
  isLoading: boolean;
  chatMessages: ChatMessage[];
  onSendChatMessage: (text: string) => void;
  isChatLoading: boolean;
  onDraftCV: () => void;
  viewMode: ViewMode;
  generatedCV: string | null;
  onBackToAnalysis: () => void;
}

const MainContent: React.FC<MainContentProps> = ({ 
  logs, 
  result, 
  isLoading, 
  chatMessages, 
  onSendChatMessage,
  isChatLoading,
  onDraftCV,
  viewMode,
  generatedCV,
  onBackToAnalysis
}) => {
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [inputMessage, setInputMessage] = React.useState('');

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, viewMode]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    onSendChatMessage(inputMessage);
    setInputMessage('');
  };

  const markdownToHtml = (markdown: string): string => {
    let html = markdown
      .replace(/^# (.*$)/gm, '<h1 style="font-size: 24pt; font-family: Arial, sans-serif; color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">$1</h1>')
      .replace(/^## (.*$)/gm, '<h2 style="font-size: 16pt; font-family: Arial, sans-serif; color: #334155; margin-top: 20px; margin-bottom: 10px; text-transform: uppercase;">$1</h2>')
      .replace(/^### (.*$)/gm, '<h3 style="font-size: 13pt; font-family: Arial, sans-serif; color: #475569; margin-top: 15px; margin-bottom: 5px; font-weight: bold;">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/^\s*-\s+(.*)$/gm, '<li style="margin-bottom: 5px;">$1</li>');

    html = html.replace(/(<li.*<\/li>)/s, '<ul>$1</ul>'); 

    const lines = html.split('\n');
    const processedLines = lines.map(line => {
      if (line.match(/<h[1-6]|<li|<ul|<\/ul/)) return line;
      if (line.trim() === '') return '<br/>';
      return `<p style="font-family: Arial, sans-serif; font-size: 11pt; line-height: 1.5; margin-bottom: 10px;">${line}</p>`;
    });

    return processedLines.join('\n');
  };

  const downloadWordDoc = () => {
    if (!generatedCV) return;
    const htmlContent = markdownToHtml(generatedCV);
    const docContent = `
      <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
      <head>
        <meta charset="utf-8">
        <title>Proposed CV</title>
        <style>body { font-family: Arial, sans-serif; font-size: 11pt; }</style>
      </head>
      <body>${htmlContent}</body>
      </html>
    `;
    const blob = new Blob(['\ufeff', docContent], { type: 'application/msword' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Proposed_CV.doc';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // --- UI Components ---

  const Navbar = () => (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-30 px-6 py-3 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-3">
        <div className="bg-slate-800 p-1.5 rounded-lg">
           <Briefcase className="text-white" size={20} />
        </div>
        <span className="text-xl font-bold text-slate-800 tracking-tight">AI Job Search Assistant</span>
      </div>
      <div className="hidden md:flex items-center space-x-8 text-sm font-medium text-slate-600">
        <button className="flex items-center gap-2 hover:text-slate-900 transition-colors"><Home size={16} /> Home</button>
        <button className="flex items-center gap-2 hover:text-slate-900 transition-colors"><Users size={16} /> Profiles</button>
        <button className="flex items-center gap-2 text-slate-900 font-semibold"><Briefcase size={16} /> Analysis</button>
        <button className="flex items-center gap-2 hover:text-slate-900 transition-colors"><Bot size={16} /> AI Chat</button>
      </div>
      <div className="flex items-center gap-3">
         <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500">
            <User size={16} />
         </div>
      </div>
    </nav>
  );

  const HeroSection = () => (
    <div className="bg-slate-900 text-white pt-12 pb-24 px-6 text-center relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full opacity-10 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-700 via-slate-900 to-slate-900"></div>
      <div className="relative z-10 max-w-3xl mx-auto">
        <div className="flex items-center justify-center gap-3 mb-6">
           <div className="p-3 bg-white/5 rounded-full backdrop-blur-sm border border-white/10">
              <Activity size={32} className="text-slate-300" />
           </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight text-white">
          Intelligent Career Matching
        </h1>
        <p className="text-slate-400 text-lg md:text-xl font-light mb-8">
          Analyze your resume against job descriptions using Agentic AI to find skill gaps and optimization opportunities.
        </p>
        
        {/* Only show "Get Started" prompt if nothing is happening */}
        {!result && !isLoading && logs.length === 0 && (
           <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-slate-400">
              <ArrowLeft size={16} className="animate-pulse" />
              Use the sidebar to upload your resume
           </div>
        )}
      </div>
    </div>
  );

  // --- View: CV Preview ---
  if (viewMode === 'cv_preview' && generatedCV) {
    return (
      <div className="flex-1 h-full overflow-hidden bg-slate-50 flex flex-col">
          <Navbar />
          {/* Toolbar */}
          <div className="bg-white border-b border-slate-200 px-6 py-3 flex justify-between items-center shadow-sm z-20">
             <div className="flex items-center gap-4">
                <button 
                  onClick={onBackToAnalysis}
                  className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
                >
                  <ArrowLeft size={16} /> Back to Dashboard
                </button>
             </div>
             <div className="flex gap-3">
               <button 
                  onClick={() => navigator.clipboard.writeText(generatedCV)}
                  className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
               >
                  <Copy size={16} />
                  <span className="hidden sm:inline">Copy Markdown</span>
               </button>
               <button 
                  onClick={downloadWordDoc}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white rounded-lg text-sm font-medium hover:bg-slate-900 transition-colors shadow-sm"
               >
                  <Download size={16} />
                  <span>Download .DOC</span>
               </button>
             </div>
          </div>

          <div className="flex-1 overflow-hidden p-6">
            <div className="max-w-[1600px] mx-auto h-full grid grid-cols-1 lg:grid-cols-12 gap-6">
               <div className="lg:col-span-7 h-full flex flex-col">
                  <div className="bg-white rounded-lg shadow-xl border border-slate-200 flex-1 overflow-y-auto w-full max-w-[210mm] mx-auto relative custom-scrollbar">
                      <div className="p-[40px] md:p-[60px] min-h-full bg-white text-slate-800">
                         <div className="prose prose-slate max-w-none prose-headings:font-sans prose-p:text-slate-700 prose-li:text-slate-700">
                           <div className="whitespace-pre-wrap font-sans text-[11pt] leading-relaxed">
                              {generatedCV.split('\n').map((line, i) => {
                                 if (line.startsWith('# ')) return <h1 key={i} className="text-3xl font-bold border-b-2 border-slate-800 pb-2 mb-4 mt-2 text-slate-900">{line.replace('# ', '')}</h1>;
                                 if (line.startsWith('## ')) return <h2 key={i} className="text-xl font-bold text-slate-800 uppercase mt-6 mb-2 tracking-wide">{line.replace('## ', '')}</h2>;
                                 if (line.startsWith('### ')) return <h3 key={i} className="text-lg font-bold text-slate-700 mt-4 mb-1">{line.replace('### ', '')}</h3>;
                                 if (line.startsWith('- ')) return <div key={i} className="flex gap-2 mb-1 ml-1"><span className="text-slate-400">•</span><span>{line.replace('- ', '').replace(/\*\*(.*?)\*\*/g, '$1')}</span></div>;
                                 return <p key={i} className="mb-2 min-h-[1em]">{line.replace(/\*\*(.*?)\*\*/g, '$1')}</p>;
                              })}
                           </div>
                         </div>
                      </div>
                  </div>
               </div>
               <div className="lg:col-span-5 h-full flex flex-col">
                  <ChatInterface 
                    chatMessages={chatMessages}
                    isChatLoading={isChatLoading}
                    inputMessage={inputMessage}
                    setInputMessage={setInputMessage}
                    onSendMessage={handleSendMessage}
                    viewMode={viewMode}
                    chatEndRef={chatEndRef}
                    heightClass="flex-1"
                  />
               </div>
            </div>
          </div>
      </div>
    );
  }

  // --- View: Dashboard (Analysis) ---
  return (
    <div className="flex-1 h-full overflow-y-auto bg-slate-50 flex flex-col">
      <Navbar />
      <HeroSection />

      <div className="flex-1 max-w-7xl mx-auto w-full px-6 -mt-16 pb-12 z-20 relative">
        
        {/* Logs Overlay - Grey Theme */}
        {(isLoading || (!result && logs.length > 0)) && (
          <div className="bg-slate-800 rounded-xl shadow-2xl border border-slate-700 p-6 mb-8 animate-fade-in text-slate-200">
             <div className="flex items-center gap-3 mb-4 border-b border-slate-700 pb-4">
                <div className={`p-2 rounded-lg bg-slate-700 text-slate-300`}>
                   <Terminal size={20} className={isLoading ? "animate-spin" : ""} />
                </div>
                <h3 className="font-semibold text-white tracking-wide">System Activity Log</h3>
             </div>
            <div className="space-y-3 max-h-60 overflow-y-auto custom-scrollbar bg-slate-900/50 p-4 rounded-lg border border-slate-700/50 font-mono text-xs">
              {logs.map((log) => (
                <div 
                  key={log.id}
                  className={`
                    flex items-center gap-3
                    ${log.type === 'success' ? 'text-green-400' : ''}
                    ${log.type === 'agent' ? 'text-blue-300' : ''}
                    ${log.type === 'error' ? 'text-red-400' : ''}
                    ${log.type === 'info' ? 'text-slate-500' : ''}
                  `}
                >
                   <span className="opacity-40 w-16">{new Date(log.timestamp).toLocaleTimeString([], {minute:'2-digit', second:'2-digit'})}</span>
                   <span>{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {result && !isLoading && (
          <div className="animate-slide-up space-y-6">
            
            <div className="flex items-center gap-2 mb-2">
                <div className="bg-slate-200 p-2 rounded-lg">
                   <BarChart2 className="text-slate-700" size={20} />
                </div>
                <h2 className="text-xl font-bold text-slate-800">Analysis Dashboard</h2>
                <p className="text-sm text-slate-500 ml-2">Key performance indicators and metrics</p>
            </div>

            {/* Metrics Row - Grey Theme */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
               {/* Match Score Card */}
               <div className="bg-white rounded-xl border border-slate-300 p-8 shadow-sm flex flex-col items-center justify-center text-center hover:shadow-md transition-shadow">
                  <div className="mb-4 p-4 bg-slate-100 rounded-full">
                     <Activity className="text-slate-700" size={40} />
                  </div>
                  <h3 className="text-5xl font-extrabold text-slate-900 mb-2">{result.matchScore}%</h3>
                  <p className="text-lg font-semibold text-slate-600">Match Score</p>
                  <p className="text-sm text-slate-400 mt-1">Alignment with Job Description</p>
               </div>

               {/* Stats Card */}
               <div className="bg-white rounded-xl border border-slate-300 p-8 shadow-sm flex flex-col items-center justify-center text-center hover:shadow-md transition-shadow">
                   <div className="mb-4 p-4 bg-slate-100 rounded-full">
                     <Users className="text-slate-700" size={40} />
                  </div>
                  <div className="flex gap-8">
                     <div>
                        <h3 className="text-4xl font-extrabold text-slate-900 mb-1">{result.candidateSkills.length}</h3>
                        <p className="text-sm font-semibold text-slate-600">Matching Skills</p>
                     </div>
                     <div className="w-px bg-slate-200"></div>
                     <div>
                        <h3 className="text-4xl font-extrabold text-slate-900 mb-1">{result.gapAnalysis.length}</h3>
                        <p className="text-sm font-semibold text-slate-600">Critical Gaps</p>
                     </div>
                  </div>
               </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
               {/* Left Column: Details */}
               <div className="lg:col-span-2 space-y-6">
                  {/* Executive Summary - Grey Theme */}
                  <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                     <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
                        <h3 className="font-bold text-slate-700 flex items-center gap-2">
                           <FileText size={18} className="text-slate-500" /> Executive Summary
                        </h3>
                     </div>
                     <div className="p-6">
                        <p className="text-slate-700 leading-relaxed">{result.overallFitSummary}</p>
                     </div>
                  </div>

                  {/* Skills & Gaps - Grey Theme */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                     <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                        <div className="px-6 py-3 border-b border-slate-200 bg-slate-50">
                           <h3 className="font-bold text-slate-700 text-sm">Skills Match</h3>
                        </div>
                        <div className="p-4 flex flex-wrap gap-2">
                           {result.candidateSkills.map((s, i) => (
                              <span key={i} className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200">{s}</span>
                           ))}
                        </div>
                     </div>
                     <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                         <div className="px-6 py-3 border-b border-slate-200 bg-slate-50">
                           <h3 className="font-bold text-slate-700 text-sm">Missing Requirements</h3>
                        </div>
                        <div className="p-4 space-y-2">
                           {result.gapAnalysis.map((g, i) => (
                              <div key={i} className="flex items-start gap-2 text-sm text-slate-700">
                                 <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0"></div>
                                 {g}
                              </div>
                           ))}
                        </div>
                     </div>
                  </div>
               </div>

               {/* Right Column: Chat & Actions */}
               <div className="space-y-6">
                  <div className="bg-slate-800 rounded-xl p-6 text-white shadow-lg border border-slate-700">
                     <h3 className="text-xl font-bold mb-2">Next Steps</h3>
                     <p className="text-slate-400 text-sm mb-6">Ready to apply? Generate a tailored resume now.</p>
                     <button
                        onClick={onDraftCV}
                        className="w-full py-3 px-4 bg-white text-slate-900 rounded-lg font-bold hover:bg-slate-200 transition-colors shadow-md flex items-center justify-center gap-2"
                     >
                        <FileText size={18} className="text-slate-700" />
                        Draft Proposed CV
                     </button>
                  </div>

                  <ChatInterface 
                    chatMessages={chatMessages}
                    isChatLoading={isChatLoading}
                    inputMessage={inputMessage}
                    setInputMessage={setInputMessage}
                    onSendMessage={handleSendMessage}
                    viewMode={viewMode}
                    chatEndRef={chatEndRef}
                  />
               </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MainContent;