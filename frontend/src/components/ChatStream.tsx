import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box } from '@mui/material'; // Example using MUI

// Define a type for our chat messages
interface ChatMessage {
  id: string;
  isBot: boolean;
  text?: string; // Make text optional for different event types
  type: 'start' | 'progress' | 'plan' | 'tool_result' | 'analysis' | 'strategy' | 'feedback' | 'error' | 'end' | 'user_query';
  // Add specific fields based on type
  steps?: string[]; // For plan
  tool_name?: string; // For tool_result
  result?: any; // For tool_result, analysis
  proposals?: any; // For strategy
  isPlanning?: boolean; // Flag for planning progress
  isSufficient?: boolean; // For analysis
  state_id?: string; // For feedback
  message?: string; // For error, end, simple progress
}

interface ChatStreamProps {
  apiEndpoint: string; // e.g., /api/invoke or /api/resume
  payload: any; // The initial data to send (user_query, user_profile, state_id etc.)
  onStreamEnd?: (threadId: string | null) => void; // Callback when stream ends
}

const ChatStream: React.FC<ChatStreamProps> = ({ apiEndpoint, payload, onStreamEnd }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null); // Ref to scroll to bottom

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]); // Scroll whenever messages update

  useEffect(() => {
    // Clear previous state on new invocation/resume
    setMessages([]);
    setError(null);
    setIsLoading(true);

    // Add the initial user query to the chat display
    if (payload.user_query) {
        setMessages([{
            id: `user-${Date.now()}`,
            isBot: false,
            text: payload.user_query,
            type: 'user_query'
        }]);
    }


    const eventSource = new EventSource(apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add any other necessary headers like Authorization if needed
      },
      body: JSON.stringify(payload),
      withCredentials: true, // Important if using cookies/sessions
    } as any); // Cast to any to allow custom body/method for EventSource polyfill or specific server setups

    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('SSE Connection Opened');
      // setIsLoading(false); // Wait for first message instead?
    };

    eventSource.onerror = (err) => {
      console.error('SSE Error:', err);
      setError('Connection error. Please try again.');
      setIsLoading(false);
      eventSource.close();
    };

    // --- Event Listeners ---

    eventSource.addEventListener('start', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.log('Start Event:', eventData);
            // Maybe add a specific start message or just wait for profile/planning
            setIsLoading(false); // Connection established, show first progress messages
        } catch (e) { console.error("Error parsing start event:", e); }
    });

    eventSource.addEventListener('progress', (event: MessageEvent) => {
      try {
        const eventData = JSON.parse(event.data);
        console.log('Progress Event:', eventData);

        // Check for the specific planning message
        if (eventData.message === "Agent planning research steps...") {
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              id: `progress-planning-${Date.now()}`,
              isBot: true,
              isPlanning: true, // Custom flag
              text: eventData.message,
              type: 'progress',
            },
          ]);
        } else {
          // Handle other progress messages if needed (e.g., update status indicator)
          console.log("Received other progress update:", eventData.message);
        }
      } catch (e) { console.error("Error parsing progress event:", e); }
    });

    eventSource.addEventListener('plan', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.log('Plan Event:', eventData);
            setMessages((prevMessages) => [
                ...prevMessages,
                { id: `plan-${Date.now()}`, isBot: true, steps: eventData.steps, type: 'plan' },
            ]);
        } catch (e) { console.error("Error parsing plan event:", e); }
    });

    // --- Listener for custom search scrape start event ---
    eventSource.addEventListener('search_scrape_start', (event: MessageEvent) => {
      try {
        const eventData = JSON.parse(event.data);
        console.log('Search Scrape Start Event:', eventData);
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: `search-start-${Date.now()}`,
            isBot: true,
            text: `Searching and scraping top results for: \"${eventData.query}\" ...`,
            type: 'progress', // Or a custom type like 'search_start' if you want specific styling
          },
        ]);
      } catch (e) { console.error("Error parsing search_scrape_start event:", e); }
    });
    // --- End listener ---

     eventSource.addEventListener('tool_result', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.log('Tool Result Event:', eventData);
            setMessages((prevMessages) => [
                ...prevMessages,
                {
                    id: `tool-${eventData.tool_name}-${Date.now()}`,
                    isBot: true,
                    tool_name: eventData.tool_name,
                    result: eventData.result, // Contains the simplified preview/error
                    type: 'tool_result'
                },
            ]);
        } catch (e) { console.error("Error parsing tool_result event:", e); }
    });

    // Add listeners for other custom events: 'analysis', 'strategy', 'feedback', 'error', 'replan', 'processing', 'storage', 'storage_ts' etc.
     eventSource.addEventListener('analysis', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.log('Analysis Event:', eventData);
            setMessages((prevMessages) => [
                ...prevMessages,
                {
                    id: `analysis-${Date.now()}`,
                    isBot: true,
                    result: eventData.result,
                    isSufficient: eventData.is_sufficient,
                    type: 'analysis'
                },
            ]);
        } catch (e) { console.error("Error parsing analysis event:", e); }
    });

    eventSource.addEventListener('strategy', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.log('Strategy Event:', eventData);
             setMessages((prevMessages) => [
                ...prevMessages,
                {
                    id: `strategy-${Date.now()}`,
                    isBot: true,
                    proposals: eventData.proposals,
                    type: 'strategy'
                },
            ]);
        } catch (e) { console.error("Error parsing strategy event:", e); }
    });

     eventSource.addEventListener('feedback', (event: MessageEvent) => {
         try {
            const eventData = JSON.parse(event.data);
            console.log('Feedback Event:', eventData);
             setMessages((prevMessages) => [
                ...prevMessages,
                {
                    id: `feedback-${Date.now()}`,
                    isBot: true,
                    message: eventData.message, // "Pipeline paused..."
                    state_id: eventData.state_id,
                    type: 'feedback'
                },
            ]);
             // Potentially trigger UI for user feedback input here
             setIsLoading(false); // Stop loading indicator as we await feedback
         } catch (e) { console.error("Error parsing feedback event:", e); }
    });

    eventSource.addEventListener('error', (event: MessageEvent) => {
        try {
            const eventData = JSON.parse(event.data);
            console.error('SSE Error Event:', eventData);
            setMessages((prevMessages) => [
                ...prevMessages,
                { id: `error-${Date.now()}`, isBot: true, message: eventData.message, type: 'error' },
            ]);
             setError(eventData.message || 'An unknown error occurred during processing.');
             setIsLoading(false);
        } catch (e) { console.error("Error parsing error event:", e); }
        // Close connection on explicit error event? Maybe depends on the error type.
        // eventSource.close();
    });

    eventSource.addEventListener('end', (event: MessageEvent) => {
      try {
          const eventData = JSON.parse(event.data);
          console.log('End Event:', eventData);
          setMessages((prevMessages) => [
            ...prevMessages,
            { id: `end-${Date.now()}`, isBot: true, message: 'Agent processing finished.', type: 'end' },
          ]);
          setIsLoading(false);
          eventSource.close();
          if (onStreamEnd) {
            onStreamEnd(eventData.thread_id || null);
          }
      } catch (e) { console.error("Error parsing end event:", e); }
    });

    // Cleanup function: close the connection when the component unmounts
    // or when dependencies change causing the effect to re-run.
    return () => {
      console.log('Closing SSE Connection');
      if (eventSourceRef.current) {
         eventSourceRef.current.close();
         eventSourceRef.current = null;
      }
      setIsLoading(false);
    };
  }, [apiEndpoint, payload, onStreamEnd]); // Re-run effect if endpoint or payload changes

  // --- Render Logic ---
  const renderMessageContent = (msg: ChatMessage) => {
    switch (msg.type) {
      case 'user_query':
        return <Typography variant="body1"><strong>You:</strong> {msg.text}</Typography>;
      case 'progress':
        return <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>{msg.text}</Typography>;
      case 'plan':
        return (
          <div>
            <Typography variant="body2" sx={{ mb: 1 }}><strong>Research Plan:</strong></Typography>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              {msg.steps?.map((step, index) => <li key={index}><Typography variant="body2">{step}</Typography></li>)}
            </ul>
          </div>
        );
      case 'tool_result':
        return (
          <div>
            <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 'bold' }}> 
              Tool: {msg.tool_name || 'Unknown'}
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
              Tool Result:
            </Typography>
            <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', bgcolor: '#f5f5f5', p: 1, borderRadius: 1, mt: 0.5 }}>
              {typeof msg.result === 'object' ? JSON.stringify(msg.result, null, 2) : String(msg.result)}
            </Typography>
          </div>
        );
       case 'analysis':
         return (
           <div>
            <Typography variant="body2" sx={{ mb: 1 }}><strong>Analysis:</strong></Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>{msg.result}</Typography>
            <Typography variant="caption" sx={{ color: msg.isSufficient ? 'green' : 'orange' }}>
                Data Sufficient: {msg.isSufficient ? 'Yes' : 'No'}
            </Typography>
           </div>
         );
        case 'strategy':
           return (
             <div>
              <Typography variant="body2" sx={{ mb: 1 }}><strong>Strategy Proposal(s):</strong></Typography>
              {/* Render proposals appropriately - assuming it's an array or object */}
              <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {JSON.stringify(msg.proposals, null, 2)}
              </Typography>
             </div>
           );
        case 'feedback':
            return (
                 <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'info.main' }}>
                    {msg.message} (State ID: {msg.state_id}) - Awaiting user feedback...
                 </Typography>
            );
      case 'error':
        return <Typography variant="body2" sx={{ color: 'error.main' }}><strong>Error:</strong> {msg.message}</Typography>;
      case 'end':
         return <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'success.main' }}>{msg.message}</Typography>;
      default:
        return <Typography variant="body1">{msg.text}</Typography>;
    }
  };

  return (
    <Box sx={{ width: '100%', height: '400px', overflowY: 'auto', border: '1px solid #ccc', p: 2, mb: 2 }}>
      {messages.map((msg) => (
        <Card
          key={msg.id}
          variant="outlined"
          sx={{
            mb: 1,
            ml: msg.isBot ? 0 : 'auto', // Align user messages right? Or just style differently
            mr: msg.isBot ? 'auto' : 0,
            maxWidth: '85%',
            bgcolor: msg.isBot ? '#f0f0f0' : 'primary.lighter', // Example different background
             ...(msg.type === 'error' && { bgcolor: 'error.lighter' }) // Specific error styling
          }}
        >
          <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}> {/* Adjust padding */}
            {renderMessageContent(msg)}
          </CardContent>
        </Card>
      ))}
      {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
              <Typography sx={{ ml: 1 }}>Connecting...</Typography>
          </Box>
      )}
      {error && (
        <Typography color="error" sx={{ mt: 1 }}>{error}</Typography>
      )}
       {/* Element to scroll to */}
       <div ref={messagesEndRef} />
    </Box>
  );
};

export default ChatStream; 