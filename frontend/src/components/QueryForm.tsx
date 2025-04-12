'use client'; // Add this for client-side interactivity

import React, { useState } from 'react';
import { Loader2 } from 'lucide-react'; // Import loader icon

// Define a basic structure for the profile data
interface UserProfileData {
    risk_tolerance?: 'low' | 'medium' | 'high' | ''; // Add empty string for default
    preferred_chains?: string;
    // Add other fields as needed
    [key: string]: any;
}

interface QueryFormProps {
  // Update onSubmitQuery to accept profile data
  onSubmitQuery: (query: string, profile: UserProfileData) => void; 
  isLoading: boolean;
}

const QueryForm: React.FC<QueryFormProps> = ({ onSubmitQuery, isLoading }) => {
  const [query, setQuery] = useState<string>('');
  // Add state for profile fields
  const [profile, setProfile] = useState<UserProfileData>({ 
      risk_tolerance: '',
      preferred_chains: '' 
  });

  const handleProfileChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
      const { name, value } = event.target;
      setProfile(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim() || isLoading) return;
    // Pass query and profile data
    onSubmitQuery(query, profile);
  };

  return (
    // Add spacing for profile fields
    <form onSubmit={handleSubmit} className="mb-4 w-full space-y-4">
      <div> {/* Wrap query input */} 
        <label htmlFor="queryInput" className="block text-sm font-medium text-gray-700 mb-1">
          Enter your research query:
        </label>
        <textarea
          id="queryInput"
          name="query" // Add name for consistency, though not used in profile
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Analyze the latest developments in liquid restaking tokens (LRTs) and propose potential strategies..."
          rows={4}
          className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100"
          disabled={isLoading}
        />
      </div>

      {/* User Profile Section */}
      <fieldset className="border border-gray-300 p-4 rounded-md">
          <legend className="text-sm font-medium text-gray-700 px-1">Optional: User Profile</legend>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <label htmlFor="risk_tolerance" className="block text-sm font-medium text-gray-700 mb-1">
                    Risk Tolerance
                </label>
                <select
                    id="risk_tolerance"
                    name="risk_tolerance"
                    value={profile.risk_tolerance}
                    onChange={handleProfileChange}
                    className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 bg-white"
                    disabled={isLoading}
                >
                    <option value="">Select...</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                </select>
            </div>
            <div>
                <label htmlFor="preferred_chains" className="block text-sm font-medium text-gray-700 mb-1">
                    Preferred Chains (comma-separated)
                </label>
                <input
                    type="text"
                    id="preferred_chains"
                    name="preferred_chains"
                    value={profile.preferred_chains}
                    onChange={handleProfileChange}
                    placeholder="e.g., Ethereum, Arbitrum, Solana"
                    className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100"
                    disabled={isLoading}
                />
            </div>
          </div>
      </fieldset>

      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="mt-2 inline-flex items-center justify-center px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Processing...
          </>
        ) : (
          'Start Research'
        )}
      </button>
    </form>
  );
};

export default QueryForm;
