import { useEffect, useMemo, useRef, useState } from 'react';
import { CandlestickSeries, ColorType, createChart, createSeriesMarkers, type IChartApi, type ISeriesApi, type ISeriesMarkersPluginApi, type Time, type UTCTimestamp } from 'lightweight-charts';
import type { ActualCandle, SessionPrediction } from '../types';
import { formatTime, price } from '../utils/format';

type HoverDetails = { x: number; y: number; candle: ActualCandle; prediction?: SessionPrediction } | null;

export function ActualChart({ candles, predictions, height = 500 }: { candles: ActualCandle[]; predictions: SessionPrediction[]; height?: number }) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markers = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const levelLines = useRef<ReturnType<ISeriesApi<'Candlestick'>['createPriceLine']>[]>([]);
  const fittedRange = useRef('');
  const predictionByTime = useMemo(() => new Map(predictions.map((prediction) => [
    Math.floor(new Date(prediction.interval_start).getTime() / 1000), prediction,
  ])), [predictions]);
  const candlesRef = useRef(candles);
  const predictionByTimeRef = useRef(predictionByTime);
  const [hover, setHover] = useState<HoverDetails>(null);
  candlesRef.current = candles;
  predictionByTimeRef.current = predictionByTime;

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const instance = createChart(node, {
      width: Math.max(1, node.clientWidth), height, autoSize: false,
      layout: { background: { type: ColorType.Solid, color: '#0b1422' }, textColor: '#9cafc5', fontFamily: 'IBM Plex Mono, monospace' },
      grid: { vertLines: { color: '#18283b' }, horzLines: { color: '#18283b' } },
      rightPriceScale: { borderColor: '#233750', scaleMargins: { top: 0.14, bottom: 0.12 } },
      timeScale: { borderColor: '#233750', timeVisible: true, secondsVisible: false, rightOffset: 2, barSpacing: 12 },
      crosshair: { vertLine: { color: '#6b85a5', labelBackgroundColor: '#213653' }, horzLine: { color: '#6b85a5', labelBackgroundColor: '#213653' } },
      localization: { timeFormatter: (time: number) => `${new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true }).format(new Date(time * 1000))} ET` },
    });
    const candleSeries = instance.addSeries(CandlestickSeries, {
      upColor: '#39dba1', downColor: '#ff697b', borderVisible: false, wickUpColor: '#39dba1', wickDownColor: '#ff697b',
    });
    chart.current = instance;
    series.current = candleSeries;
    markers.current = createSeriesMarkers(candleSeries, []);
    instance.subscribeCrosshairMove((param) => {
      if (!param.point || param.time == null) { setHover(null); return; }
      const candle = param.seriesData.get(candleSeries) as { open?: number; high?: number; low?: number; close?: number } | undefined;
      if (!candle) { setHover(null); return; }
      const time = Number(param.time);
      const actual = candlesRef.current.find((bar) => Math.floor(new Date(bar.time).getTime() / 1000) === time);
      if (!actual) { setHover(null); return; }
      setHover({ x: param.point.x, y: param.point.y, candle: actual, prediction: predictionByTimeRef.current.get(time) });
    });
    const resize = () => instance.applyOptions({ width: Math.max(1, node.clientWidth), height });
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    resize();
    return () => { observer.disconnect(); markers.current?.detach(); instance.remove(); chart.current = null; series.current = null; markers.current = null; };
  }, [height]);

  useEffect(() => {
    const candleSeries = series.current;
    const instance = chart.current;
    if (!candleSeries || !instance) return;
    const points = candles.map((bar) => ({
      time: Math.floor(new Date(bar.time).getTime() / 1000) as UTCTimestamp,
      open: bar.open, high: bar.high, low: bar.low, close: bar.close,
    }));
    candleSeries.setData(points);
    for (const line of levelLines.current) candleSeries.removePriceLine(line);
    levelLines.current = [];
    const referencePrediction = predictions[0];
    if (referencePrediction?.support != null) levelLines.current.push(candleSeries.createPriceLine({ price: referencePrediction.support, color: '#39dba1', lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: 'SUPPORT' }));
    if (referencePrediction?.resistance != null) levelLines.current.push(candleSeries.createPriceLine({ price: referencePrediction.resistance, color: '#ff697b', lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: 'RESISTANCE' }));
    if (referencePrediction?.reference_price != null) levelLines.current.push(candleSeries.createPriceLine({ price: referencePrediction.reference_price, color: '#9bb2ce', lineStyle: 3, lineWidth: 1, axisLabelVisible: true, title: 'REFERENCE' }));
    const times = new Set(points.map((p) => p.time));
    markers.current?.setMarkers(predictions.filter((p) => times.has(Math.floor(new Date(p.interval_start).getTime() / 1000) as UTCTimestamp)).map((p) => ({
      time: Math.floor(new Date(p.interval_start).getTime() / 1000) as UTCTimestamp,
      position: p.predicted_signal === 'UP' ? 'belowBar' : 'aboveBar',
      color: p.predicted_signal === 'UP' ? '#39dba1' : '#ff697b',
      shape: p.predicted_signal === 'UP' ? 'arrowUp' : 'arrowDown',
      text: p.predicted_signal === 'UP' ? '↑ UP' : '↓ DOWN',
      size: 1.2,
    })));
    const rangeSignature = points.length ? `${points[0].time}:${points.at(-1)?.time}:${points.length}` : '';
    if (rangeSignature && rangeSignature !== fittedRange.current) {
      instance.timeScale().fitContent();
      fittedRange.current = rangeSignature;
    }
  }, [candles, predictions]);

  return <div className="actual-chart-wrap">
    <div className="actual-chart" ref={host} style={{ height }} aria-label="Actual AMZN ten-minute candles with UP and DOWN prediction labels" />
    {hover && <div className="chart-hover-card" style={{ left: Math.min(Math.max(hover.x + 16, 12), Math.max(12, (host.current?.clientWidth || 400) - 275)), top: Math.min(Math.max(hover.y + 12, 8), height - 225) }}>
      <div className="hover-title">{hover.prediction ? <><b className={hover.prediction.predicted_signal.toLowerCase()}>{hover.prediction.predicted_signal === 'UP' ? '↑ UP' : '↓ DOWN'}</b><span>{hover.prediction.actual?.outcome || 'PENDING'}</span></> : <b>AMZN · ACTUAL 10M</b>}</div>
      {hover.prediction && <>
        <div><span>Prediction timestamp</span><b>{formatTime(hover.prediction.interval_start, true)}</b></div>
        <div><span>Generated at</span><b>{formatTime(hover.prediction.generated_at, true)}</b></div>
        <div><span>Forecast interval</span><b>{formatTime(hover.prediction.interval_start)}–{formatTime(hover.prediction.interval_end)}</b></div>
        <div><span>Raw model score · P(UP)</span><b>{(hover.prediction.raw_probability_up * 100).toFixed(1)}%</b></div>
        <div><span>Actual outcome</span><b>{hover.prediction.actual?.actual_direction || 'Pending'}</b></div>
        <div><span>Prediction result</span><b className={hover.prediction.actual?.outcome === 'CORRECT' ? 'correct' : hover.prediction.actual?.outcome === 'INCORRECT' ? 'incorrect' : 'pending'}>{hover.prediction.actual?.outcome || 'PENDING'}</b></div>
      </>}
      <div className="hover-ohlc"><span>O {price(hover.candle.open)}</span><span>H {price(hover.candle.high)}</span><span>L {price(hover.candle.low)}</span><span>C {price(hover.candle.close)}</span><span>VOL {hover.candle.volume.toLocaleString()}</span></div>
    </div>}
  </div>;
}
