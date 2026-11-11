// Values copied from reports/experiments/trading_analysis_engine_audit/replay_metadata.json
// and regime_performance.csv. They are a frozen historical research snapshot, not live metrics.
export const researchPerformance = {
  period: { start: '2025-10-06', end: '2026-11-09' },
  source: 'Local completed AMZN RTH 5-minute snapshot · 21,420 candles',
  validAnalyses: 17_786,
  signalCounts: { BUY: 578, SELL: 595, HOLD: 1_800, WAIT: 14_813 },
  strategies: [
    { name: 'Combined engine', setups: 1_173, resolved: 810, grossReturn: 12.5222, netReturn: -68.4778, maxDrawdown: 70.8297, grossProfitFactor: 1.1060, netProfitFactor: 0.5985, avgNetReturn: -0.0845, wins: 271 },
    { name: 'EMA20 / EMA50', setups: 2_423, resolved: 1_575, grossReturn: 8.6059, netReturn: -148.8941, maxDrawdown: 150.7550, grossProfitFactor: 1.0357, netProfitFactor: 0.5698, avgNetReturn: -0.0945 },
    { name: 'RSI 14 extremes', setups: 784, resolved: 629, grossReturn: 16.4666, netReturn: -46.4334, maxDrawdown: 54.4818, grossProfitFactor: 1.1350, netProfitFactor: 0.7132, avgNetReturn: -0.0738 },
    { name: 'Frozen ML direction only', setups: 2_240, resolved: 1_410, grossReturn: 95.9591, netReturn: -45.0409, maxDrawdown: 60.8781, grossProfitFactor: 1.5232, netProfitFactor: 0.8318, avgNetReturn: -0.0319 },
  ],
  regimes: [
    { name: 'Trending up', candles: 2_046, setups: 125, resolved: 94, netReturn: -12.8558, maxDrawdown: 13.4966 },
    { name: 'Trending down', candles: 2_207, setups: 150, resolved: 100, netReturn: -10.0182, maxDrawdown: 11.4530 },
    { name: 'Sideways', candles: 3_906, setups: 328, resolved: 222, netReturn: -17.3431, maxDrawdown: 17.5312 },
    { name: 'High volatility', candles: 2_981, setups: 189, resolved: 166, netReturn: -7.9734, maxDrawdown: 12.0196 },
    { name: 'Low volatility', candles: 3_908, setups: 381, resolved: 228, netReturn: -20.2873, maxDrawdown: 21.1034 },
    { name: 'Transitional', candles: 2_738, setups: 0, resolved: 0, netReturn: 0, maxDrawdown: 0 },
  ],
  assumptions: {
    transactionCostBpsRoundTrip: 10,
    positionSize: 'One share, fixed size; additive returns (not compounded)',
    entry: 'Next same-session 5-minute open only when inside the hypothetical entry zone',
    exit: 'Stop/target touch; stop-first if both barriers are touched; otherwise session end or 180-minute limit',
    target1: 'Informational milestone only; no partial exit is booked',
  },
} as const;
