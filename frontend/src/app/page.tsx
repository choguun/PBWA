'use client';

import React from 'react';
import QueryForm from '@/components/QueryForm';
import StreamDisplay from '@/components/StreamDisplay';
import FeedbackPrompt from '@/components/FeedbackPrompt';
import { useResearchAgent } from '@/hooks/useResearchAgent'; // Import the custom hook
import { Loader2, XCircle, CheckCircle } from 'lucide-react';

// Define expected event data structures (match backend/schemas.py if possible)
interface BaseEventData {
  type?: string;
  message?: string;
  [key: string]: any; // Allow other fields
}

interface FeedbackEventData extends BaseEventData {
  type: 'awaiting_feedback';
  state_id: string;
}

interface StrategyEventData extends BaseEventData {
  type: 'strategy';
  proposals: string[];
}

type AppStatus = 'idle' | 'running' | 'paused' | 'resuming' | 'finished' | 'error';

interface LogEntry {
  eventType: string;
  data: BaseEventData;
}

export default function Home() {
  // Use the custom hook to manage state and logic
  const {
    status,
    streamLog,
    currentProposals,
    errorMessage,
    isLoading,
    handleInvokeQuery,
    handleResumeWithFeedback,
  } = useResearchAgent();

  // State ID is managed within the hook, only needed for conditional rendering check
  const isPaused = status === 'paused';

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 lg:p-12 bg-gray-100">
      <div className="z-10 w-full max-w-4xl items-center font-sans text-sm flex flex-col gap-6">
        <h1 className="text-2xl md:text-4xl font-bold text-center mb-2 text-gray-900">
          DeFi Deep Research Agent
        </h1>

        {/* Conditional Rendering for QueryForm / Status */}
        {status === 'idle' || status === 'finished' || status === 'error' ? (
            <QueryForm onSubmitQuery={handleInvokeQuery} isLoading={isLoading} />
        ) : (
            <div className="w-full text-center p-4 rounded-md bg-indigo-50 border border-indigo-200">
                 <p className="text-lg font-semibold text-indigo-700">
                    {isLoading && <Loader2 className="inline-block mr-2 h-5 w-5 animate-spin align-middle" />}
                    {status === 'running' && 'Agent is running...'}
                    {status === 'resuming' && 'Agent is resuming...'}
                    {status === 'paused' && 'Agent is paused, awaiting feedback below.'}
                 </p>
            </div>
        )}

        {errorMessage && (
          <div className="w-full p-4 bg-red-100 border border-red-300 text-red-800 rounded-md shadow-sm">
            <div className="flex items-center">
                 <XCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                 <span className="font-medium">Error:</span>
            </div>
            <p className="ml-7 text-sm">{errorMessage}</p>
          </div>
        )}

        {/* Stream Display: Render whenever agent is active or finished */} 
        {(status === 'running' || status === 'resuming' || status === 'paused' || status === 'finished') && (
          <StreamDisplay streamLog={streamLog} />
        )}

        {isPaused && (
          <FeedbackPrompt
            proposals={currentProposals}
            onSubmitFeedback={handleResumeWithFeedback}
            isLoading={isLoading}
          />
        )}

        {status === 'finished' && (
             <div className="w-full p-4 bg-green-100 border border-green-300 text-green-800 rounded-md shadow-sm text-center">
                 <div className="flex items-center justify-center">
                     <CheckCircle className="h-5 w-5 mr-2" />
                     <span className="font-medium">Agent run finished successfully.</span>
                 </div>
             </div>
        )}
      </div>
    </main>
  );
}
