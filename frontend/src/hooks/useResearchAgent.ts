'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

// Re-define types within the hook or import from a shared types file
interface BaseEventData {
  type?: string;
  message?: string;
  [key: string]: any;
}

interface FeedbackEventData extends BaseEventData {
  type: 'awaiting_feedback';
  state_id: string;
}

interface StrategyEventData extends BaseEventData {
  type: 'strategy';
  proposals: string | string[];
}

type AppStatus = 'idle' | 'running' | 'paused' | 'resuming' | 'finished' | 'error';

interface LogEntry {
  eventType: string;
  data: BaseEventData;
}

// Define UserProfileData within the hook or import from shared types
interface UserProfileData {
    risk_tolerance?: 'low' | 'medium' | 'high' | '';
    preferred_chains?: string;
    [key: string]: any;
}

interface UseResearchAgentReturn {
  status: AppStatus;
  streamLog: LogEntry[];
  currentProposals: string[];
  errorMessage: string | null;
  isLoading: boolean;
  handleInvokeQuery: (query: string, profile: UserProfileData) => Promise<void>;
  handleResumeWithFeedback: (feedback: string) => Promise<void>;
}

export function useResearchAgent(): UseResearchAgentReturn {
  const [status, setStatus] = useState<AppStatus>('idle');
  const [streamLog, setStreamLog] = useState<LogEntry[]>([]);
  const [currentProposals, setCurrentProposals] = useState<string[]>([]);
  const [stateId, setStateId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Use a ref to track the reader/cancellation capability
  const readerRef = useRef<{ cancel: () => Promise<void> } | null>(null);

  const isLoading = status === 'running' || status === 'resuming';

  // Helper function to process the stream
  const processStream = useCallback(async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      currentStatus: AppStatus // Pass current status to avoid stale closures
  ) => {
    const decoder = new TextDecoder();
    let buffer = '';
    let eventType = 'message'; // Keep track of event type outside loop
    let eventCount = 0; // DEBUG: Count events

    // Assign cancellation ability to the ref
    readerRef.current = { cancel: () => reader.cancel() };

    console.log(`[${currentStatus}] processStream: Starting loop...`);
    while (true) {
      try {
        console.log(`[${currentStatus}] processStream: Waiting for reader.read()...`);
        const { done, value } = await reader.read();
        console.log(`[${currentStatus}] processStream: reader.read() returned - done: ${done}`);

        if (done) {
          console.log(`[${currentStatus}] processStream: Stream finished by reader.`);
          setStatus(prevStatus => {
            if (prevStatus !== 'paused' && prevStatus !== 'error') {
              return 'finished';
            }
            return prevStatus;
          });
          readerRef.current = null;
          break;
        }

        const decodedChunk = decoder.decode(value, { stream: true });
        console.log(`[${currentStatus}] processStream: Decoded chunk:`, decodedChunk);
        buffer += decodedChunk;
        console.log(`[${currentStatus}] processStream: Buffer before split: "${buffer}"`);
        const messageBlocks = buffer.split('\n\n');
        console.log(`[${currentStatus}] processStream: Split into ${messageBlocks.length} blocks.`);
        buffer = messageBlocks.pop() || ''; // Keep partial message
        console.log(`[${currentStatus}] processStream: Buffer after split: "${buffer}"`);

        let shouldBreakLoop = false;
        messageBlocks.forEach((block, blockIndex) => {
          if (shouldBreakLoop || !block.trim()) return;
          eventCount++;
          console.log(`[${currentStatus}] processStream: Processing Block ${blockIndex + 1} (Event ${eventCount}):`, block);
          let currentBlockEventType = 'message'; // Reset for each block
          let eventData = '';
          block.split('\n').forEach(line => {
            if (line.startsWith('event:')) {
              currentBlockEventType = line.substring(6).trim();
              eventType = currentBlockEventType; // Update outer scope for break message
            } else if (line.startsWith('data:')) {
              eventData = line.substring(5).trim();
            }
          });
          console.log(`[${currentStatus}] processStream: Parsed - Type: ${currentBlockEventType}, Data: "${eventData}"`);

          if (eventData) {
            try {
              const parsedData = JSON.parse(eventData);
              const logEntry: LogEntry = { eventType: currentBlockEventType, data: parsedData };
              console.log(`[${currentStatus}] processStream: Calling setStreamLog for Event ${eventCount}...`);
              setStreamLog(prev => [...prev, logEntry]);
              console.log(`[${currentStatus}] processStream: setStreamLog finished for Event ${eventCount}.`);

              if (currentBlockEventType === 'strategy') {
                const proposals = (parsedData as StrategyEventData).proposals;
                setCurrentProposals(Array.isArray(proposals) ? proposals : (typeof proposals === 'string' ? proposals.split('\n').filter(p => p.trim() !== '') : []));
              }
              if (currentBlockEventType === 'feedback') {
                console.log(`[${currentStatus}] processStream: FEEDBACK event detected.`);
                setStateId((parsedData as FeedbackEventData).state_id);
                setStatus('paused');
                shouldBreakLoop = true;
              }
              if (currentBlockEventType === 'end') {
                console.log(`[${currentStatus}] processStream: END event detected.`);
                setStatus('finished');
                shouldBreakLoop = true;
              }
            } catch (e) {
              console.error(`[${currentStatus}] processStream: Failed to parse message data:`, eventData, e);
              setStreamLog(prev => [...prev, { eventType: 'parse_error', data: { message: `Invalid JSON: ${eventData}` } }]);
            }
          }
        }); // End forEach block

        if (shouldBreakLoop) {
          console.log(`[${currentStatus}] processStream: Breaking stream loop due to event: ${eventType}`);
          if (readerRef.current) {
            readerRef.current.cancel().catch(e => console.warn("Error cancelling reader on break:", e));
            readerRef.current = null;
          }
          break;
        }
      } catch (error: any) {
         console.error(`[${currentStatus}] processStream: Error during reader.read() or processing:`, error);
         if (error.name === 'AbortError' || (error instanceof DOMException && error.message.includes('aborted'))) {
           console.log(`[${currentStatus}] processStream: Stream reading aborted expectedly.`);
         } else {
           setErrorMessage(error.message || 'Stream processing failed');
           setStatus('error');
         }
         readerRef.current = null;
         break;
       }
    } // End while(true)
    console.log(`[${currentStatus}] processStream: Exited loop.`);
  }, []); // No dependencies needed

  // Handler for submitting the initial query
  const handleInvokeQuery = useCallback(async (query: string, profile: UserProfileData) => {
    console.log("handleInvokeQuery called with profile:", profile);
    setStreamLog([]);
    setErrorMessage(null);
    setCurrentProposals([]);
    setStateId(null);
    // Cancel any existing stream reader before starting new one
    if (readerRef.current) {
      console.log("Cancelling existing reader before invoke...");
      await readerRef.current.cancel().catch(e => console.warn("Error cancelling existing reader:", e));
      readerRef.current = null;
    }
    setStatus('running');

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

    // Clean up profile data: remove empty string values
    const cleanedProfile: Record<string, any> = {};
    for (const [key, value] of Object.entries(profile)) {
        if (value !== '' && value !== null && value !== undefined) {
            cleanedProfile[key] = value;
        }
    }

    try {
      const response = await fetch(`${backendUrl}/invoke`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        // Include cleaned profile data in the body
        body: JSON.stringify({ user_query: query, user_profile: cleanedProfile }), 
      });

      if (!response.ok) {
        const errorText = await response.text();
        // Try parsing error detail from text
        let detail = errorText;
        try {
            const jsonError = JSON.parse(errorText);
            detail = jsonError.detail || errorText;
        } catch {}
        throw new Error(`Backend error: ${response.status} ${detail}`);
      }
      if (!response.body) {
        throw new Error('Response body is null');
      }

      await processStream(response.body.getReader(), 'running');

    } catch (error: any) {
        // Catch errors from the initial fetch call itself or stream processing
        if (error.name === 'AbortError') {
            console.log("Fetch aborted (likely intentional cancellation).");
        } else {
            console.error('Invoke error:', error);
            // Error message already contains backend details from the throw above
            setErrorMessage(error.message || 'Failed to invoke agent');
            setStatus('error');
        }
        readerRef.current = null; // Ensure ref is cleared on fetch error
    }
  }, [processStream]); // Depend on processStream

  // Handler for resuming after feedback
  const handleResumeWithFeedback = useCallback(async (feedback: string) => {
    if (!stateId) {
      setErrorMessage('Cannot resume without state ID');
      setStatus('error');
      return;
    }

    console.log("handleResumeWithFeedback called");
    setErrorMessage(null);
    // No need to cancel reader, should already be null after pause
    if (readerRef.current) {
        console.warn("Reader ref was not null before resume? Attempting cancel...");
        await readerRef.current.cancel().catch(e => console.warn("Error cancelling reader before resume:", e));
        readerRef.current = null;
    }
    setStatus('resuming');

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${backendUrl}/resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ thread_id: stateId, feedback: feedback }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error: ${response.status} ${errorText || response.statusText}`);
      }
      if (!response.body) {
        throw new Error('Response body is null');
      }

      await processStream(response.body.getReader(), 'resuming');

    } catch (error: any) {
        if (error.name === 'AbortError') {
            console.log("Resume fetch aborted.");
        } else {
            console.error('Resume error:', error);
             // Attempt to parse backend error details (similar to invoke)
             let detailedMessage = 'Failed to resume agent';
             if (error instanceof Error) {
                 detailedMessage = error.message;
             }
             if (error && error.message && error.message.includes('Backend error:')) {
                 try {
                     const match = error.message.match(/\{.*\}/);
                     if (match && match[0]) {
                         const errorJson = JSON.parse(match[0]);
                         detailedMessage = errorJson.detail || error.message;
                     } else {
                          detailedMessage = error.message;
                     }
                 } catch (parseError) {
                     detailedMessage = error.message; // Fallback
                 }
             }
            setErrorMessage(detailedMessage);
            setStatus('error');
        }
        readerRef.current = null;
    }
  }, [stateId, processStream]); // Depend on stateId and processStream

  // Effect for cleanup (optional, as handlers should manage cancellation)
  useEffect(() => {
      return () => {
          if (readerRef.current) {
              console.log("Component unmounting, cancelling active reader...");
              readerRef.current.cancel().catch(e => console.warn("Error cancelling reader on unmount:", e));
              readerRef.current = null;
          }
      };
  }, []);

  return {
    status,
    streamLog,
    currentProposals,
    errorMessage,
    isLoading,
    handleInvokeQuery,
    handleResumeWithFeedback,
  };
} 