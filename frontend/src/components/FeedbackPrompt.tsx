'use client';

import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface FeedbackPromptProps {
  proposals: string[]; // The strategy proposals received
  onSubmitFeedback: (feedback: string) => void;
  isLoading: boolean; // To disable form during resume processing
}

const FeedbackPrompt: React.FC<FeedbackPromptProps> = ({ proposals, onSubmitFeedback, isLoading }) => {
  const [feedback, setFeedback] = useState<string>('');

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isLoading) return;
    onSubmitFeedback(feedback); // Submit feedback (even if empty)
  };

  return (
    <div className="mt-6 p-4 border border-amber-300 rounded bg-amber-50 w-full shadow">
      <h3 className="text-lg font-semibold text-amber-800 mb-2">Feedback Required:</h3>
      <p className="text-sm text-amber-700 mb-3">The agent has proposed the following strategies and paused. Review them and provide feedback or confirm to continue.</p>

      <div className="mb-4">
        <h4 className="font-semibold text-sm mb-1">Proposed Strategies:</h4>
        {proposals && proposals.length > 0 ? (
          <ul className="list-disc list-inside pl-2 text-sm space-y-1">
            {proposals.map((proposal, index) => (
              <li key={index}>{proposal}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-500 italic">No proposals received.</p>
        )}
      </div>

      <form onSubmit={handleSubmit} className="mt-4">
        <label htmlFor="feedbackInput" className="block text-sm font-medium text-gray-700 mb-1">
          Your Feedback (Optional):
        </label>
        <textarea
          id="feedbackInput"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="e.g., Prefer strategy 1, but explore token X yield first..."
          rows={3}
          className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading}
          className="mt-2 inline-flex items-center justify-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Resuming...
            </>
          ) : (
            'Submit Feedback & Continue'
          )}
        </button>
      </form>
    </div>
  );
};

export default FeedbackPrompt;
