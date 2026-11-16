import { TerminalDashboard } from './components/terminal/TerminalDashboard';

export default function App() {
  return (
    <div className="root-terminal-app">
      <header className="global-terminal-nav">
        <div className="global-brand">
          <span className="global-brand-mark">A</span>
          <span>
            <b>AMZN DECISION TERMINAL</b>
            <small>AI trading analysis · institutional decision terminal</small>
          </span>
        </div>
        <nav aria-label="Application sections">
          <button className="active">Terminal</button>
        </nav>
        <span className="global-mode">PREDICTION ONLY · NO ORDER EXECUTION</span>
      </header>
      <TerminalDashboard />
    </div>
  );
}
