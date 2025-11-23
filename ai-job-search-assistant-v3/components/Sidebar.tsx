import React, { useState } from 'react';
import { HelpCircle, FileText, X, UploadCloud, Link } from 'lucide-react';

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
    <div className="w-full md:w-80 lg:w-96 bg-white border-r border-gray-200 flex flex-col h-full p-6 shrink-0 overflow-y-auto shadow-sm z-20">
      <h2 className="text-xl font-bold text-gray-900 mb-8 flex items-center gap-2">
        <span className="w-1 h-6 bg-green-500 rounded-full"></span>
        Inputs
      </h2>

      {/* Job URL Input */}
      <div className="mb-6">
        <label htmlFor="jobUrl" className="block text-sm font-semibold text-gray-700 mb-2">
          Job Description URL
        </label>
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Link size={16} className="text-gray-400" />
          </div>
          <input
            id="jobUrl"
            type="url"
            value={jobUrl}
            onChange={(e) => setJobUrl(e.target.value)}
            placeholder="https://www.linkedin.com/jobs/..."
            className="w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent text-sm transition-all shadow-sm"
          />
        </div>
      </div>

      {/* File Upload */}
      <div className="mb-8">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
           Resume (PDF)
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
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 flex flex-col items-center justify-center bg-gray-50 group-hover:bg-green-50 group-hover:border-green-300 transition-all text-center h-40">
               <div className="bg-white p-3 rounded-full shadow-sm mb-3 group-hover:scale-110 transition-transform duration-300">
                  <UploadCloud className="text-green-600" size={24} />
               </div>
              <p className="text-sm font-medium text-gray-900 mb-1">Click to upload</p>
              <p className="text-xs text-gray-500">PDF up to 200MB</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center p-3 bg-white border border-green-200 rounded-xl shadow-sm relative group overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-green-500"></div>
            <div className="bg-green-50 p-2 rounded-lg mr-3">
               <FileText className="text-green-600" size={20} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {resumeFile.name}
              </p>
              <p className="text-xs text-gray-500">
                {(resumeFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <button
              onClick={removeFile}
              className="p-1.5 hover:bg-red-50 hover:text-red-500 rounded-full text-gray-400 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Action Button */}
      <button
        onClick={onRunAnalysis}
        disabled={isLoading || !jobUrl || !resumeFile}
        className={`w-full py-3 px-4 rounded-xl font-bold text-sm shadow-md transition-all transform active:scale-[0.98]
          ${
            isLoading || !jobUrl || !resumeFile
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200 shadow-none'
              : 'bg-gray-900 text-white hover:bg-gray-800 hover:shadow-lg'
          }
        `}
      >
        {isLoading ? 'Processing...' : 'Run Analysis'}
      </button>
      
      <div className="mt-auto pt-6 border-t border-gray-100">
        <p className="text-xs text-center text-gray-400 font-medium">Powered by Gemini 2.5</p>
      </div>
    </div>
  );
};

export default Sidebar;
