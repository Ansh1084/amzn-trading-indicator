import type { SessionResult } from '../types';

export function SessionDecision({ result }: { result: SessionResult | null }) {
  const decision = result?.final_decision;
  return <section className="panel decision-panel">
    <div className="section-heading compact-heading"><div><span className="eyebrow">SESSION SUMMARY</span><h2>Final decision</h2></div>
      <span className={`signal-badge ${decision?.decision?.toLowerCase() || 'pending'}`}>{decision?.decision || 'PENDING'}</span></div>
    <div className="vote-row"><div><small>UP VOTES</small><b className="up-copy">{decision?.up_count ?? '—'}</b></div><div><small>DOWN VOTES</small><b className="down-copy">{decision?.down_count ?? '—'}</b></div><div><small>NEUTRAL</small><b>{decision?.neutral_count ?? '—'}</b></div></div>
    <div className="actual-outcome"><span>ACTUAL SESSION OUTCOME</span><b>{result?.actual_session?.decision || 'Pending evaluation'}</b></div>
    <p className="small-note">Predicted vote and actual session outcome are separate. Aggregate probability is not shown because the model score is uncalibrated.</p>
  </section>;
}
