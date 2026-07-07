import { useEffect, useState } from 'react';

function App() {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/api/messages`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setMessages(data.data);
        }
      });
  }, []);

  return (
    <div>
      <h1>My Portfolio</h1>

      <h2>Messages</h2>
      {messages.map((msg, index) => (
        <div key={index}>
          <p><strong>{msg.name}</strong>: {msg.message}</p>
        </div>
      ))}
    </div>
  );
}

export default App;
