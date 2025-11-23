import React, { useState } from 'react';
import { HelpCircle, FileText, X, UploadCloud, Link, ArrowRight } from 'lucide-react';

interface SidebarProps {
  jobUrl: string;
  setJobUrl: (url: string) => void;
  resumeFile: File | null;
  setResumeFile: (file: File | null) => void;
  onRunAnalysis: () => void;
  isLoading: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  jobUrl,
  setJobUrl,
  resumeFile,
  setResumeFile,
  onRunAnalysis,
  isLoading
}) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setResumeFile(e.target.files[0]);
    }
  };

  const removeFile = () => {
    setResumeFile(null);
  };

  return (
    <div className="w-full md:w-80 lg:w-96 bg-white border-r border-slate-200 flex flex-col h-full shrink-0 overflow-y-auto shadow-[4px_0_24px_rgba(0,0,0,0.02)] z-30 relative">
      <div className="p-8 pb-4">
        <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Configuration</h2>
        <p className="text-sm text-slate-500 mt-1">Setup your analysis parameters</p>
      </div>

      <div className="px-8 space-y-8">
        {/* Job URL Input */}
        <div>
          <label htmlFor="jobUrl" className="block text-sm font-bold text-slate-700 mb-2">
            Target Job URL
          </label>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Link size={16} className="text-slate-400 group-focus-within:text-slate-700 transition-colors" />
            </div>
            <input
              id="jobUrl"
              type="url"
              value={jobUrl}
              onChange={(e) => setJobUrl(e.target.value)}
              placeholder="https://linkedin.com/jobs/..."
              className="w-full pl-10 pr-3 py-3 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-500/20 focus:border-slate-500 text-sm transition-all text-slate-800 placeholder-slate-400"
            />
          </div>
        </div>

        {/* File Upload */}
        <div>
          <label className="block text-sm font-bold text-slate-700 mb-2">
             Resume / CV
          </label>

          {!resumeFile ? (
            <div className="relative group">
              <input
                type="file"
                id="resume-upload"
                accept="application/pdf"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className="border-2 border-dashed border-slate-200 rounded-xl p-8 flex flex-col items-center justify-center bg-slate-50 group-hover:bg-slate-100 group-hover:border-slate-400 transition-all text-center h-48">
                 <div className="bg-white p-3 rounded-full shadow-sm mb-4 group-hover:scale-110 transition-transform duration-300 border border-slate-100">
                    <UploadCloud className="text-slate-600" size={24} />
                 </div>
                <p className="text-sm font-semibold text-slate-900 mb-1">Upload PDF Resume</p>
                <p className="text-xs text-slate-500">Drag & drop or click to browse</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center p-4 bg-slate-50 border border-slate-200 rounded-xl relative group">
              <div className="bg-white p-2.5 rounded-lg mr-3 shadow-sm border border-slate-100">
                 <FileText className="text-slate-600" size={20} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-slate-800 truncate">
                  {resumeFile.name}
                </p>
                <p className="text-xs text-slate-500">
                  {(resumeFile.size / 1024).toFixed(1)} KB • PDF
                </p>
              </div>
              <button
                onClick={removeFile}
                className="p-1.5 hover:bg-red-50 hover:text-red-500 rounded-full text-slate-400 transition-colors"
              >
                <X size={18} />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="mt-auto p-8 border-t border-slate-100 bg-slate-50/50">
        <button
          onClick={onRunAnalysis}
          disabled={isLoading || !jobUrl || !resumeFile}
          className={`w-full py-4 px-6 rounded-xl font-bold text-sm shadow-lg transition-all transform active:scale-[0.98] flex items-center justify-center gap-2
            ${
              isLoading || !jobUrl || !resumeFile
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                : 'bg-slate-900 text-white hover:bg-slate-800 hover:shadow-xl hover:-translate-y-0.5'
            }
          `}
        >
          {isLoading ? (
            <>Processing...</>
          ) : (
            <>Run Analysis <ArrowRight size={16} /></>
          )}
        </button>
      </div>
    </div>
  );
};

export default Sidebar;