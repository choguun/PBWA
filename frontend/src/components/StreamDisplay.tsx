'use client';

import React from 'react';
import {
  Rocket, FileText, Wrench, RefreshCw, Hourglass, Database, BarChart, Activity, 
  Lightbulb, Sparkles, MessageSquare, CheckCircle, XCircle, AlertTriangle, Info
} from 'lucide-react';

// Define a base structure and potentially more specific types based on backend schemas
interface BaseEventData {
  type?: string;
  message?: string;
  steps?: string[];
  result?: any;
  new_plan?: string[];
  is_sufficient?: boolean;
  proposals?: string | string[]; // Strategy proposals might be string or list
  [key: string]: any;
}

interface LogEntry {
  eventType: string;
  data: BaseEventData;
}

interface StreamDisplayProps {
  streamLog: LogEntry[];
}

// Helper to get styling based on event type
const getEventStyle = (eventType: string): { Icon: React.ElementType; color: string; bgColor: string } => {
  const iconProps = { size: 18, className: 'inline-block mr-2 flex-shrink-0' }; // Common icon props
  switch (eventType) {
    case 'start': return { Icon: () => <Rocket {...iconProps} />, color: 'text-blue-800', bgColor: 'bg-blue-50' };
    case 'plan': return { Icon: () => <FileText {...iconProps} />, color: 'text-indigo-800', bgColor: 'bg-indigo-50' };
    case 'tool_result': return { Icon: () => <Wrench {...iconProps} />, color: 'text-gray-700', bgColor: 'bg-gray-100' };
    case 'replan': return { Icon: () => <RefreshCw {...iconProps} />, color: 'text-cyan-800', bgColor: 'bg-cyan-50' };
    case 'processing': return { Icon: () => <Hourglass {...iconProps} />, color: 'text-gray-700', bgColor: 'bg-gray-50' };
    case 'storage': return { Icon: () => <Database {...iconProps} />, color: 'text-lime-800', bgColor: 'bg-lime-50' };
    case 'storage_ts': return { Icon: () => <BarChart {...iconProps} />, color: 'text-teal-800', bgColor: 'bg-teal-50' };
    case 'analysis': return { Icon: () => <Activity {...iconProps} />, color: 'text-purple-800', bgColor: 'bg-purple-50' };
    case 'strategy': return { Icon: () => <Lightbulb {...iconProps} />, color: 'text-pink-800', bgColor: 'bg-pink-50' };
    case 'refinement': return { Icon: () => <Sparkles {...iconProps} />, color: 'text-green-800', bgColor: 'bg-green-50' };
    case 'feedback': return { Icon: () => <MessageSquare {...iconProps} />, color: 'text-amber-800', bgColor: 'bg-amber-50' };
    case 'end': return { Icon: () => <CheckCircle {...iconProps} />, color: 'text-green-800', bgColor: 'bg-green-100' };
    case 'error': return { Icon: () => <XCircle {...iconProps} />, color: 'text-red-800', bgColor: 'bg-red-100' };
    case 'parse_error': return { Icon: () => <AlertTriangle {...iconProps} />, color: 'text-red-700', bgColor: 'bg-red-50' };
    default: return { Icon: () => <Info {...iconProps} />, color: 'text-gray-700', bgColor: 'bg-gray-50' };
  }
};

const renderEventData = (entry: LogEntry) => {
  const { eventType, data } = entry;
  const { Icon, color, bgColor } = getEventStyle(eventType);

  switch (eventType) {
    case 'start':
      return <p>Query received. Starting research...</p>;
    case 'plan':
      return (
        <div>
          <p className="font-medium">Generated Plan:</p>
          {data.steps && data.steps.length > 0 ? (
            <ol className="list-decimal list-inside pl-2 space-y-1 text-sm">
              {data.steps.map((step, i) => <li key={i}>{step}</li>)}
            </ol>
          ) : (
            <p className="text-gray-500 italic text-sm">Empty plan received.</p>
          )}
        </div>
      );
    case 'tool_result':
      return (
        <div>
          <p className="font-medium">Tool Result:</p>
          {/* Make pre tag scrollable if content overflows */}
          <pre className="whitespace-pre-wrap text-xs bg-white p-2 rounded mt-1 border border-gray-200 max-h-40 overflow-auto">
            {JSON.stringify(data.result, null, 2)}
          </pre>
        </div>
      );
    case 'replan':
      return (
        <div>
          <p className={`font-medium ${color}`}>Replanning...</p>
          {data.new_plan && (
            <div className="mt-1">
              <p className="font-medium text-sm">New Plan:</p>
              <ol className="list-decimal list-inside pl-2 space-y-1 text-sm">
                {data.new_plan.map((step, i) => <li key={i}>{step}</li>)}
              </ol>
            </div>
          )}
        </div>
      );
    case 'processing':
    case 'storage':
    case 'storage_ts':
    case 'refinement':
    case 'progress':
       return <p className="text-sm">{data.message || `Processing event: ${eventType}`}</p>;
    case 'analysis':
      return (
        <div>
          <p className="font-medium">Analysis Results:</p>
          <p className="text-sm">Sufficient: {data.is_sufficient ? <span className="text-green-600 font-medium">Yes</span> : <span className="text-red-600 font-medium">No</span>}</p>
          <pre className="whitespace-pre-wrap text-xs bg-white p-2 rounded mt-1 border border-gray-200 max-h-60 overflow-auto">
            {data.result || "No analysis text provided."}
          </pre>
          {data.reasoning && <p className="text-xs text-gray-600 mt-1">Reason: {data.reasoning}</p>}
          {data.suggestions_for_next_steps && !data.is_sufficient && <p className="text-xs text-blue-600 mt-1">Suggestion: {data.suggestions_for_next_steps}</p>}
        </div>
      );
    case 'strategy':
       const proposals = Array.isArray(data.proposals) ? data.proposals : (typeof data.proposals === 'string' ? data.proposals.split('\n').filter(p => p.trim() !== '') : []);
       return (
         <div>
           <p className={`font-medium ${color}`}>Proposed Strategies:</p>
           {proposals && proposals.length > 0 ? (
             <ul className="list-disc list-inside pl-2 space-y-1 text-sm">
                {proposals.map((proposal, i) => <li key={i}>{proposal}</li>)}
             </ul>
           ) : (
             <p className="text-gray-500 italic text-sm">No specific proposals parsed.</p>
           )}
         </div>
       );
     case 'awaiting_feedback': // Should be handled by 'feedback' event type now
     case 'feedback':
        return <p className={`font-medium ${color}`}>{data.message || "Awaiting user feedback..."}</p>;
    case 'error':
      return <p className={`font-medium ${color}`}>Error: {data.message || 'An unknown error occurred.'}</p>;
    case 'end':
        return <p className={`font-medium ${color}`}>{data.message || "Stream ended."}</p>;
    case 'parse_error':
        return <p className={`font-medium ${color}`}>Stream Data Parse Error: {data.message || 'Could not parse data.'}</p>;
    default:
      // Render unrecognized events with default styling but show the data
      return (
        <div>
            <p className="font-medium">Unknown Event ({eventType}):</p>
            <pre className="whitespace-pre-wrap text-xs bg-white p-2 rounded mt-1 border border-gray-200 max-h-40 overflow-auto">
                {JSON.stringify(data, null, 2)}
            </pre>
        </div>
      );
  }
};

const StreamDisplay: React.FC<StreamDisplayProps> = ({ streamLog }) => {
  const scrollRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [streamLog]);

  return (
    <div ref={scrollRef} className="w-full mt-4 p-4 border border-gray-300 rounded-lg shadow-inner bg-white h-[60vh] overflow-y-auto text-sm">
      <h3 className="text-lg font-semibold mb-3 sticky top-0 bg-white pb-2 border-b border-gray-200 z-10">Agent Progress:</h3>
      {streamLog.length === 0 ? (
        <p className="text-gray-500 italic px-2">Waiting for agent to start...</p>
      ) : (
        <ul className="space-y-3 px-2">
          {streamLog.map((logEntry, index) => {
             const { Icon, color, bgColor } = getEventStyle(logEntry.eventType);
             return (
                <li key={index} className={`p-3 rounded-md border ${bgColor} ${color}`}>
                    <div className="flex items-start mb-1"> {/* Use items-start for alignment */}
                        <Icon /> {/* Render the icon component */} 
                        <span className="font-mono text-xs font-medium uppercase tracking-wider mt-px"> {/* Adjust alignment */} 
                            {logEntry.eventType.replace('_', ' ')}
                        </span>
                    </div>
                    <div className="pl-7"> {/* Indent content further to align with text */} 
                        {renderEventData(logEntry)}
                    </div>
                </li>
             )
          })}
        </ul>
      )}
    </div>
  );
};

export default StreamDisplay;
