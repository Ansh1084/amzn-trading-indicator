import React from 'react';
import { createRoot } from 'react-dom/client';
function App() { return <main style={{fontFamily:'sans-serif',padding:24}}><h1>AMZN Trading Indicator</h1><p>Frontend scaffold — chart and signals will be implemented later.</p></main>; }
createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
