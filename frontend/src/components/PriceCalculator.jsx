// In your PriceCalculator.jsx
import { useState } from 'react';
import axios from 'axios';

export default function PriceCalculator() {
  const [connectionStatus, setConnectionStatus] = useState('');
  
  // Add this test function
  const testBackendConnection = async () => {
    try {
      const response = await axios.get('http://localhost:5000/test-connection');
      setConnectionStatus(response.data.message);
    } catch (error) {
      setConnectionStatus(`Connection failed: ${error.message}`);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      {/* Add a test button at the top */}
      <div className="mb-8">
        <button 
          onClick={testBackendConnection}
          className="btn btn-sm btn-info"
        >
          Test Backend Connection
        </button>
        {connectionStatus && (
          <div className="mt-2 text-sm">
            Status: <span className="font-bold">{connectionStatus}</span>
          </div>
        )}
      </div>

      {/* Your existing form goes here */}
    </div>
  );
}