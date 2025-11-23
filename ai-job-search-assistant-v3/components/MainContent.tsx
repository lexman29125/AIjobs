import React, { useRef, useEffect } from 'react';
import { Search, CheckCircle, Activity, AlertCircle, Briefcase, UserCheck, ArrowRight, MessageSquare, Send, FileText, ArrowLeft, Copy, Download, Bot, User } from 'lucide-react';
import { AnalysisResult, LogEntry, ChatMessage, ViewMode } from '../types';

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

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, viewMode]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    onSendChatMessage(inputMessage);
    setInputMessage('');
  };

  /**
   * Converts basic Markdown to HTML for the Word export.
   * This is a simple parser specifically for the structured CV output.
   */
  const markdownToHtml = (markdown: string): string => {
    let html = markdown
      // Headers
      .replace(/^# (.*$)/gm, '<h1 style="font-size: 24pt; font-family: Arial, sans-serif; color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">$1</h1>')
      .replace(/^## (.*$)/gm, '<h2 style="font-size: 16pt; font-family: Arial, sans-serif; color: #2E5C55; margin-top: 20px; margin-bottom: 10px; text-transform: uppercase;">$1</h2>')
      .replace(/^### (.*$)/gm, '<h3 style="font-size: 13pt; font-family: Arial, sans-serif; color: #444; margin-top: 15px; margin-bottom: 5px; font-weight: bold;">$1</h3>')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Lists
      .replace(/^\s*-\s+(.*)$/gm, '<li style="margin-bottom: 5px;">$1</li>');

    // Wrap lists in <ul>
    // This regex looks for consecutive <li> lines and wraps them. 
    // Simplified approach: just wrapping the whole thing in a body and letting Word handle the list items often works, 
    // but explicit <ul> is better. For this simple export, we'll assume the browser/Word parses <li> correctly.
    // A more robust way for simple export:
    html = html.replace(/(<li.*<\/li>)/s, '<ul>$1</ul>'); // Very basic, improved below via line-by-line

    // Line breaks to paragraphs if not a header or list
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
    
    // Construct a complete HTML file with Word-specific namespaces
    const docContent = `
      <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
      <head>
        <meta charset="utf-8">
        <title>Proposed CV</title>
        <style>
          body { font-family: Arial, sans-serif; font-size: 11pt; }
        </style>
      </head>
      <body>
        ${htmlContent}
      </body>
      </html>
    `;

    const blob = new Blob(['\ufeff', docContent], {
      type: 'application/msword'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Proposed_CV.doc'; // .doc opens in Word perfectly with HTML content
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Reusable Chat Interface Component
  const renderChatInterface = (heightClass: string = "h-[400px]") => (
    <div className={`bg-white border border-gray-200 rounded-xl overflow-hidden flex flex-col ${heightClass} shadow-lg`}>
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center gap-2">
        <Bot size={18} className="text-green-600" />
        <span className="font-semibold text-gray-700 text-sm">AI Career Coach</span>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50">
          {chatMessages.length === 0 && (
             <div className="text-center text-gray-400 mt-10 text-sm">
                Start chatting to discuss the analysis...
             </div>
          )}
          {chatMessages.map((msg, idx) => (
            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'model' && (
                  <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center shrink-0 border border-green-200">
                    <Bot size={16} className="text-green-700" />
                  </div>
                )}
                <div className={`
                  px-4 py-2.5 rounded-2xl max-w-[80%] text-sm shadow-sm leading-relaxed
                  ${msg.role === 'user' 
                      ? 'bg-blue-600 text-white rounded-tr-none' 
                      : 'bg-white border border-gray-200 text-gray-800 rounded-tl-none'
                  }
                `}>
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0 border border-blue-200">
                    <User size={16} className="text-blue-700" />
                  </div>
                )}
            </div>
          ))}
          {isChatLoading && (
            <div className="flex justify-start gap-3">
                 <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center shrink-0 border border-green-200">
                    <Bot size={16} className="text-green-700" />
                  </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                  <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                  </div>
                </div>
            </div>
          )}
          <div ref={chatEndRef} />
      </div>
      <div className="bg-white border-t border-gray-200 p-3">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder={viewMode === 'cv_preview' ? "E.g., 'Make the summary more concise'..." : "Type your feedback..."}
                className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm bg-gray-50 focus:bg-white transition-colors"
                disabled={isChatLoading}
            />
            <button 
                type="submit"
                disabled={!inputMessage.trim() || isChatLoading}
                className="bg-blue-600 text-white p-2.5 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm active:transform active:scale-95"
            >
                <Send size={18} />
            </button>
          </form>
      </div>
    </div>
  );

  // View: CV Preview (Split Screen)
  if (viewMode === 'cv_preview' && generatedCV) {
    return (
      <div className="flex-1 h-full overflow-hidden bg-gray-100 flex flex-col">
          {/* Header */}
          <div className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center shadow-sm shrink-0 z-10">
             <div className="flex items-center gap-4">
                <button 
                  onClick={onBackToAnalysis}
                  className="p-2 hover:bg-gray-100 rounded-full text-gray-500 transition-colors"
                  title="Back to Analysis"
                >
                  <ArrowLeft size={20} />
                </button>
                <h2 className="font-bold text-gray-800 text-lg flex items-center gap-2">
                   <FileText className="text-green-600" size={20} />
                   CV Draft & Refinement
                </h2>
             </div>
             
             <div className="flex gap-3">
               <button 
                  onClick={() => navigator.clipboard.writeText(generatedCV)}
                  className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900 transition-colors shadow-sm"
               >
                  <Copy size={16} />
                  <span className="hidden sm:inline">Copy Markdown</span>
               </button>
               <button 
                  onClick={downloadWordDoc}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors shadow-sm"
               >
                  <Download size={16} />
                  <span>Download Word Doc</span>
               </button>
             </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-hidden p-6">
            <div className="max-w-[1600px] mx-auto h-full grid grid-cols-1 lg:grid-cols-12 gap-6">
               
               {/* Left: CV Paper View */}
               <div className="lg:col-span-7 h-full flex flex-col">
                  <div className="bg-white rounded-sm shadow-xl border border-gray-200 flex-1 overflow-y-auto w-full max-w-[210mm] mx-auto relative">
                      {/* Paper Pattern/Texture */}
                      <div className="absolute inset-0 pointer-events-none opacity-[0.02]" style={{backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '20px 20px'}}></div>
                      
                      <div className="p-[40px] md:p-[60px] min-h-full bg-white">
                         <div className="prose prose-slate max-w-none prose-headings:font-sans prose-headings:text-gray-900 prose-p:text-gray-700 prose-li:text-gray-700">
                           {/* Render simple markdown preview */}
                           <div className="whitespace-pre-wrap font-sans text-[11pt] leading-relaxed text-gray-800">
                              {generatedCV.split('\n').map((line, i) => {
                                 // Simple visual parser for the preview
                                 if (line.startsWith('# ')) return <h1 key={i} className="text-3xl font-bold border-b-2 border-gray-800 pb-2 mb-4 mt-2">{line.replace('# ', '')}</h1>;
                                 if (line.startsWith('## ')) return <h2 key={i} className="text-xl font-bold text-gray-800 uppercase mt-6 mb-2">{line.replace('## ', '')}</h2>;
                                 if (line.startsWith('### ')) return <h3 key={i} className="text-lg font-bold text-gray-700 mt-4 mb-1">{line.replace('### ', '')}</h3>;
                                 if (line.startsWith('- ')) return <div key={i} className="flex gap-2 mb-1 ml-1"><span className="text-gray-500">•</span><span>{line.replace('- ', '').replace(/\*\*(.*?)\*\*/g, '$1')}</span></div>;
                                 return <p key={i} className="mb-2 min-h-[1em]">{line.replace(/\*\*(.*?)\*\*/g, '$1')}</p>;
                              })}
                           </div>
                         </div>
                      </div>
                  </div>
                  <div className="text-center mt-2 text-xs text-gray-400">Preview Mode • ATS Optimized Layout</div>
               </div>

               {/* Right: Chat */}
               <div className="lg:col-span-5 h-full flex flex-col">
                  <div className="mb-4">
                     <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                        <Bot className="text-green-600" size={20} />
                        AI Editor
                     </h3>
                     <p className="text-sm text-gray-500">Refine the content. E.g., "Rewrite the summary to focus on leadership."</p>
                  </div>
                  {renderChatInterface("flex-1")}
               </div>
            </div>
          </div>
      </div>
    );
  }

  // View: Analysis (Standard)
  return (
    <div className="flex-1 h-full overflow-y-auto bg-gray-50/50 p-6 md:p-10">
      <div className="max-w-4xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="bg-green-100 p-3 rounded-xl">
               <Briefcase className="text-green-600" size={32} />
            </div>
          </div>
          <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight mb-2">AI Job Search Assistant</h1>
          <p className="text-gray-500 text-lg">Analyze fit, identify gaps, and generate ATS-ready resumes.</p>
        </div>

        {/* Live Logs & Status Area */}
        {(isLoading || logs.length > 0) && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 mb-8">
            <div className="space-y-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
              {logs.map((log) => (
                <div 
                  key={log.id}
                  className={`
                    flex items-start gap-3 p-2.5 rounded-lg text-sm transition-all duration-300
                    ${log.type === 'success' ? 'bg-green-50 text-green-700' : ''}
                    ${log.type === 'agent' ? 'text-purple-600 font-medium' : ''}
                    ${log.type === 'info' ? 'text-gray-600' : ''}
                  `}
                >
                  {log.type === 'success' && <CheckCircle size={16} className="mt-0.5 shrink-0" />}
                  {log.type === 'agent' && <Activity size={16} className="mt-0.5 shrink-0" />}
                  {log.type === 'info' && <AlertCircle size={16} className="mt-0.5 shrink-0 text-gray-400" />}
                  <span>{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Section */}
        {result && !isLoading && (
          <div className="animate-slide-up space-y-8">
            
            {/* Action Card */}
            <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl shadow-lg p-6 text-white flex flex-col md:flex-row items-center justify-between gap-6">
                <div>
                   <h2 className="text-2xl font-bold mb-1">Analysis Complete</h2>
                   <p className="text-green-100 opacity-90">Your profile matches <strong className="text-white">{result.matchScore}%</strong> of the job requirements.</p>
                </div>
                <button
                    onClick={onDraftCV}
                    disabled={isChatLoading}
                    className="flex items-center gap-2 px-6 py-3 bg-white text-green-700 rounded-xl hover:bg-green-50 transition-all shadow-md font-bold disabled:opacity-80 active:transform active:scale-95 whitespace-nowrap"
                  >
                    <FileText size={18} />
                    {isChatLoading ? 'Drafting...' : 'Draft Proposed CV'}
                  </button>
            </div>

            {/* Analysis Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
               {/* Summary Card */}
               <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm md:col-span-2">
                  <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
                     <Activity className="text-blue-500" size={20} />
                     Executive Summary
                  </h3>
                  <p className="text-gray-700 leading-relaxed text-[15px]">
                    {result.overallFitSummary}
                  </p>
               </div>

              {/* Candidate Skills */}
              <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                     <UserCheck className="text-blue-600" size={20} />
                     Your Strengths
                  </h3>
                  <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full text-xs font-bold">{result.candidateSkills.length} found</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {result.candidateSkills.map((skill, idx) => (
                    <span key={idx} className="inline-flex items-center px-2.5 py-1 rounded-md text-sm font-medium bg-blue-50 text-blue-700 border border-blue-100">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              {/* Required Job Skills */}
              <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm flex flex-col">
                 <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                       <Briefcase className="text-purple-600" size={20} />
                       Job Requirements
                    </h3>
                 </div>
                <div className="flex flex-wrap gap-2">
                  {result.requiredJobSkills.map((skill, idx) => (
                    <span key={idx} className="inline-flex items-center px-2.5 py-1 rounded-md text-sm font-medium bg-purple-50 text-purple-700 border border-purple-100">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Gap Analysis */}
            {result.gapAnalysis.length > 0 && (
               <div className="bg-orange-50/50 border border-orange-100 rounded-2xl p-6">
                  <div className="flex items-center gap-2 mb-4">
                     <AlertCircle className="text-orange-600" size={20} />
                     <h3 className="text-lg font-bold text-orange-900">Critical Gaps</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                     {result.gapAnalysis.map((gap, idx) => (
                        <div key={idx} className="flex items-start gap-3 bg-white p-3 rounded-lg border border-orange-100/50 shadow-sm">
                           <div className="mt-1 w-2 h-2 rounded-full bg-orange-400 shrink-0"></div>
                           <span className="text-sm text-gray-800">{gap}</span>
                        </div>
                     ))}
                  </div>
               </div>
            )}

            {/* Chat Section */}
            <div>
               <div className="mb-4 px-1">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                     <MessageSquare className="text-green-600" size={20} />
                     Strategic Discussion
                  </h3>
                  <p className="text-sm text-gray-500">Discuss these results with your AI agent before drafting the resume.</p>
               </div>
               {renderChatInterface()}
            </div>
          </div>
        )}
        
        {!result && !isLoading && logs.length === 0 && (
           <div className="flex flex-col items-center justify-center py-20 px-4 text-center border-2 border-dashed border-gray-200 rounded-3xl bg-gray-50/30">
              <div className="bg-white p-4 rounded-full shadow-sm mb-4">
                 <Search size={32} className="text-green-500" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Ready to Analyze</h3>
              <p className="text-gray-500 text-sm max-w-sm mx-auto">Upload your PDF resume and paste a LinkedIn job URL on the left to get started.</p>
           </div>
        )}

      </div>
    </div>
  );
};

export default MainContent;
