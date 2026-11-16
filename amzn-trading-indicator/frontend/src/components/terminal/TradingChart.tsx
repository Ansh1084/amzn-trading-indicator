import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, HistogramSeries, LineSeries, LineStyle,
  type IChartApi, type IPriceLine, type ISeriesApi, type SeriesMarker, type Time, type UTCTimestamp,
  type ISeriesPrimitive, type SeriesAttachedParameter, type IPrimitivePaneRenderer, type IPrimitivePaneView,
  type IChartApiBase,
} from 'lightweight-charts';
import type { CanvasRenderingTarget2D } from 'fancy-canvas';
import type { AnalysisPayload, MarketCandle, SignalRecord, Zone } from '../../types/analysis';
import type { MarketIntelligence } from '../../services/analysisApi';
import { formatTime, price } from '../../utils/format';

interface ZoneDrawingItem {
  zone: Zone;
  isSupply: boolean;
  formationTime: UTCTimestamp;
  upper: number;
  lower: number;
  status: string;
}

class SupplyDemandPaneRenderer implements IPrimitivePaneRenderer {
  private _items: ZoneDrawingItem[];
  private _latestTime: UTCTimestamp | null;
  private _latestIndex: number | null;
  private _series: ISeriesApi<'Candlestick', Time>;
  private _chart: IChartApiBase<Time>;

  constructor(
    items: ZoneDrawingItem[],
    latestTime: UTCTimestamp | null,
    latestIndex: number | null,
    series: ISeriesApi<'Candlestick', Time>,
    chart: IChartApiBase<Time>,
  ) {
    this._items = items;
    this._latestTime = latestTime;
    this._latestIndex = latestIndex;
    this._series = series;
    this._chart = chart;
  }

  draw(target: CanvasRenderingTarget2D) {
    target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      const timeScale = this._chart.timeScale();

      // Find the exact X-coordinate of the latest candle on screen.
      // A finite zone must terminate strictly at the latest candle and NEVER extend
      // into the empty future/right-margin space of the chart.
      let lastCandleX: number | null = null;
      if (this._latestIndex !== null && this._latestIndex !== undefined && this._latestIndex >= 0) {
        const coord = timeScale.logicalToCoordinate(this._latestIndex as unknown as import('lightweight-charts').Logical);
        if (coord !== null) {
          lastCandleX = Number(coord);
        }
      }

      if (lastCandleX === null && this._latestTime) {
        const barIdx = timeScale.timeToIndex(this._latestTime, true);
        if (barIdx !== null) {
          const coord = timeScale.logicalToCoordinate(barIdx as unknown as import('lightweight-charts').Logical);
          if (coord !== null) lastCandleX = Number(coord);
        }
      }

      if (lastCandleX === null && this._latestTime) {
        const coord = timeScale.timeToCoordinate(this._latestTime);
        if (coord !== null) lastCandleX = Number(coord);
      }

      // Default right edge: strictly bounded at the rightmost candle (+ 4px half-bar margin)
      const defaultX2 = lastCandleX !== null
        ? Math.max(0, Math.min(mediaSize.width, lastCandleX + 4))
        : mediaSize.width;

      for (const item of this._items) {
        // Y1, Y2: Map upper and lower zone prices to vertical coordinates
        const yUpper = this._series.priceToCoordinate(item.upper);
        const yLower = this._series.priceToCoordinate(item.lower);
        if (yUpper === null && yLower === null) continue;
        const topY = yUpper !== null ? Number(yUpper) : 0;
        const bottomY = yLower !== null ? Number(yLower) : mediaSize.height;

        const Y1 = Math.min(topY, bottomY);
        const Y2 = Math.max(topY, bottomY);
        const height = Math.max(Y2 - Y1, 3);

        // X1: Map formation timestamp to horizontal coordinate
        let X1 = 0;
        const visibleRange = timeScale.getVisibleRange();
        if (visibleRange && item.formationTime > 0 && item.formationTime > Number(visibleRange.to)) {
          // Formed in the future relative to the currently scrolled viewport
          continue;
        }

        let coord: number | null = null;
        if (item.formationTime > 0) {
          const directCoord = timeScale.timeToCoordinate(item.formationTime);
          if (directCoord !== null) {
            coord = Number(directCoord);
          } else {
            const barIdx = timeScale.timeToIndex(item.formationTime, true);
            if (barIdx !== null) {
              const logicalCoord = timeScale.logicalToCoordinate(barIdx as unknown as import('lightweight-charts').Logical);
              if (logicalCoord !== null) {
                coord = Number(logicalCoord);
              }
            }
          }
        }

        if (coord !== null) {
          X1 = Math.max(0, coord);
        } else {
          // Formed in the past before visible range, extends from the left edge
          X1 = 0;
        }

        // X2: Calculate finite right boundary
        let X2 = defaultX2;
        if (item.status === 'INVALIDATED' && item.zone.invalidation_timestamp) {
          const invalTime = epoch(item.zone.invalidation_timestamp);
          let invalCoord: number | null = null;
          const direct = timeScale.timeToCoordinate(invalTime);
          if (direct !== null) {
            invalCoord = Number(direct);
          } else {
            const barIdx = timeScale.timeToIndex(invalTime, true);
            if (barIdx !== null) {
              const logicalCoord = timeScale.logicalToCoordinate(barIdx as unknown as import('lightweight-charts').Logical);
              if (logicalCoord !== null) invalCoord = Number(logicalCoord);
            }
          }
          if (invalCoord !== null) {
            X2 = Math.max(X1, Math.min(defaultX2, invalCoord));
          }
        }

        const width = X2 - X1;
        if (width <= 0) continue;

        const isInvalidated = item.status === 'INVALIDATED';
        const isSupply = item.isSupply;

        // Visual design strictly matching reference:
        // Red translucent bands for selling-pressure areas (Supply)
        // Green translucent bands for buying-pressure areas (Demand)
        if (isSupply) {
          context.fillStyle = isInvalidated ? 'rgba(242, 54, 69, 0.05)' : 'rgba(242, 54, 69, 0.14)';
          context.strokeStyle = isInvalidated ? 'rgba(242, 54, 69, 0.25)' : 'rgba(242, 54, 69, 0.50)';
        } else {
          context.fillStyle = isInvalidated ? 'rgba(8, 153, 129, 0.05)' : 'rgba(8, 153, 129, 0.14)';
          context.strokeStyle = isInvalidated ? 'rgba(8, 153, 129, 0.25)' : 'rgba(8, 153, 129, 0.50)';
        }

        context.fillRect(X1, Y1, width, height);
        context.lineWidth = 1;
        if (isInvalidated) {
          context.setLineDash([4, 4]);
        } else {
          context.setLineDash([]);
        }
        context.strokeRect(X1, Y1, width, height);
        context.setLineDash([]);
      }
    });
  }
}

class SupplyDemandPaneView implements IPrimitivePaneView {
  private _source: SupplyDemandZonesPrimitive;

  constructor(source: SupplyDemandZonesPrimitive) {
    this._source = source;
  }

  zOrder(): 'bottom' {
    return 'bottom';
  }

  renderer(): IPrimitivePaneRenderer | null {
    if (!this._source.series || !this._source.chart) return null;
    return new SupplyDemandPaneRenderer(
      this._source.items,
      this._source.latestTime,
      this._source.latestIndex,
      this._source.series,
      this._source.chart,
    );
  }
}

class SupplyDemandZonesPrimitive implements ISeriesPrimitive<Time> {
  chart: IChartApiBase<Time> | null = null;
  series: ISeriesApi<'Candlestick', Time> | null = null;
  items: ZoneDrawingItem[] = [];
  latestTime: UTCTimestamp | null = null;
  latestIndex: number | null = null;
  private _requestUpdate?: () => void;
  private _paneViews: SupplyDemandPaneView[];

  constructor() {
    this._paneViews = [new SupplyDemandPaneView(this)];
  }

  attached(param: SeriesAttachedParameter<Time, 'Candlestick'>) {
    this.chart = param.chart;
    this.series = param.series;
    this._requestUpdate = param.requestUpdate;
  }

  detached() {
    this.chart = null;
    this.series = null;
    this._requestUpdate = undefined;
  }

  update(zones: Zone[], latestTime: UTCTimestamp | null, latestIndex: number | null = null) {
    this.latestTime = latestTime;
    this.latestIndex = latestIndex;
    // Render active and tested zones only to maintain a clean, limited, finite terminal chart
    const activeZones = zones.filter((z) => z.status !== 'INVALIDATED');
    const displayZones = activeZones.length > 0 ? activeZones : zones;
    this.items = displayZones.map((z) => {
      const isSupply = z.zone_type === 'SUPPLY' || z.type === 'RESISTANCE';
      return {
        zone: z,
        isSupply,
        formationTime: z.formation_timestamp ? epoch(z.formation_timestamp) : (0 as UTCTimestamp),
        upper: z.upper,
        lower: z.lower,
        status: z.status ?? 'ACTIVE',
      };
    });
    this._requestUpdate?.();
  }

  paneViews() {
    return this._paneViews;
  }
}

type OverlayKey = 'ema9' | 'ema20' | 'ema50' | 'sma20' | 'sma50' | 'vwap' | 'bollinger';
type OverlayState = Record<OverlayKey, boolean>;
type Hover = { x: number; y: number; candle: MarketCandle; signal?: SignalRecord } | null;
type ChartSeries = ISeriesApi<'Line'>;

const overlaySpec: Array<{ key: OverlayKey; label: string; color: string }> = [
  { key: 'ema9', label: 'EMA 9', color: '#2962ff' },
  { key: 'ema20', label: 'EMA 20', color: '#ff9800' },
  { key: 'ema50', label: 'EMA 50', color: '#20a28a' },
  { key: 'sma20', label: 'SMA 20', color: '#b97bff' },
  { key: 'sma50', label: 'SMA 50', color: '#a5a8b2' },
  { key: 'vwap', label: 'Session VWAP', color: '#e3c547' },
  { key: 'bollinger', label: 'Bollinger 20,2', color: '#8290a7' },
];

const etDate = (timestamp: string) => new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
}).format(new Date(timestamp));

const epoch = (timestamp: string) => Math.floor(new Date(timestamp).getTime() / 1000) as UTCTimestamp;

function rollingLines(candles: MarketCandle[]) {
  const closes = candles.map((c) => c.close);
  const result: Record<OverlayKey, Array<{ time: UTCTimestamp; value: number }[]>> = {
    ema9: [], ema20: [], ema50: [], sma20: [], sma50: [], vwap: [], bollinger: [],
  };
  for (const span of [9, 20, 50]) {
    const alpha = 2 / (span + 1);
    let value = closes[0] ?? 0;
    const points = closes.map((close, i) => {
      value = i === 0 ? close : alpha * close + (1 - alpha) * value;
      return i + 1 >= span && Number.isFinite(value) ? { time: epoch(candles[i].timestamp), value } : null;
    }).filter((p): p is { time: UTCTimestamp; value: number } => p !== null);
    if (span === 9) result.ema9.push(points);
    if (span === 20) result.ema20.push(points);
    if (span === 50) result.ema50.push(points);
  }
  for (const window of [20, 50]) {
    const points = closes.flatMap((_, i) => i + 1 < window ? [] : [{
      time: epoch(candles[i].timestamp),
      value: closes.slice(i + 1 - window, i + 1).reduce((sum, n) => sum + n, 0) / window,
    }]);
    if (window === 20) result.sma20.push(points);
    if (window === 50) result.sma50.push(points);
  }
  let activeDay = '';
  let cumulativePV = 0;
  let cumulativeVolume = 0;
  const vwap: Array<{ time: UTCTimestamp; value: number }> = [];
  const upper: Array<{ time: UTCTimestamp; value: number }> = [];
  const lower: Array<{ time: UTCTimestamp; value: number }> = [];
  for (let i = 0; i < candles.length; i += 1) {
    const candle = candles[i];
    const day = etDate(candle.timestamp);
    if (day !== activeDay) { activeDay = day; cumulativePV = 0; cumulativeVolume = 0; }
    const typical = (candle.high + candle.low + candle.close) / 3;
    cumulativePV += typical * candle.volume;
    cumulativeVolume += candle.volume;
    if (cumulativeVolume > 0) vwap.push({ time: epoch(candle.timestamp), value: cumulativePV / cumulativeVolume });
    if (i >= 19) {
      const sample = closes.slice(i - 19, i + 1);
      const mid = sample.reduce((sum, n) => sum + n, 0) / 20;
      const sd = Math.sqrt(sample.reduce((sum, n) => sum + (n - mid) ** 2, 0) / 20);
      upper.push({ time: epoch(candle.timestamp), value: mid + 2 * sd });
      lower.push({ time: epoch(candle.timestamp), value: mid - 2 * sd });
    }
  }
  result.vwap.push(vwap);
  result.bollinger.push(upper, lower);
  return result;
}

function niceTime(time: Time) {
  const value = typeof time === 'number' ? new Date(time * 1000) : new Date(`${time}T16:00:00-04:00`);
  const sameDay = typeof time === 'number' && new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', dateStyle: 'short' }).format(value);
  const date = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', month: 'short', day: '2-digit' }).format(value);
  const clock = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true }).format(value);
  return sameDay ? `${date} ${clock}` : `${date} ${clock}`;
}

export function TradingChart({ candles, formingCandle, signals, payload, intelligence, selectedTimestamp, onVisibleRightCandle }: {
  candles: MarketCandle[]; formingCandle?: MarketCandle | null; signals: SignalRecord[]; payload?: AnalysisPayload; intelligence?: MarketIntelligence | null; selectedTimestamp?: string | null; onVisibleRightCandle?: (candle: MarketCandle) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candleSeries = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeries = useRef<ISeriesApi<'Histogram'> | null>(null);
  const markerPlugin = useRef<ReturnType<typeof createSeriesMarkers<Time>> | null>(null);
  const zonesPlugin = useRef<SupplyDemandZonesPrimitive | null>(null);
  const overlaySeries = useRef(new Map<string, ChartSeries>());
  const priceLines = useRef<IPriceLine[]>([]);
  const rangeSignature = useRef('');
  const candlesRef = useRef(candles);
  const plottedCandlesRef = useRef<MarketCandle[]>([]);
  const zonesRef = useRef<Zone[]>([]);
  const showZonesRef = useRef(true);
  const signalsByTime = useMemo(() => new Map(signals.map((record) => [epoch(record.candle_timestamp), record])), [signals]);
  const signalsRef = useRef(signalsByTime);
  const onVisibleRightCandleRef = useRef(onVisibleRightCandle);
  const [hover, setHover] = useState<Hover>(null);
  const [hoveredZone, setHoveredZone] = useState<Zone | null>(null);
  const [overlays, setOverlays] = useState<OverlayState>({ ema9: true, ema20: true, ema50: false, sma20: false, sma50: false, vwap: true, bollinger: false });
  const [showVolume, setShowVolume] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showRisk, setShowRisk] = useState(true);
  const [showMarkers, setShowMarkers] = useState(true);
  const [showOrbLevels, setShowOrbLevels] = useState(false);
  const [showPreviousDayLevels, setShowPreviousDayLevels] = useState(false);
  const [showPreviousClose, setShowPreviousClose] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  candlesRef.current = candles;
  showZonesRef.current = showZones;
  signalsRef.current = signalsByTime;
  onVisibleRightCandleRef.current = onVisibleRightCandle;

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const instance = createChart(node, {
      width: Math.max(1, node.clientWidth), height: 560, autoSize: false,
      layout: { background: { type: ColorType.Solid, color: '#131722' }, textColor: '#787b86', fontFamily: 'Plus Jakarta Sans, sans-serif', attributionLogo: true },
      grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
      rightPriceScale: { borderColor: '#434651', scaleMargins: { top: 0.1, bottom: 0.22 } },
      timeScale: { borderColor: '#434651', timeVisible: true, secondsVisible: false, rightOffset: 6, barSpacing: 8, tickMarkFormatter: (time: Time) => niceTime(time) },
      crosshair: { vertLine: { color: '#787b86', labelBackgroundColor: '#2962ff' }, horzLine: { color: '#787b86', labelBackgroundColor: '#2962ff' } },
      localization: { timeFormatter: (time: Time) => `${niceTime(time)} ET` },
    });
    const candlesApi = instance.addSeries(CandlestickSeries, {
      upColor: '#089981', downColor: '#f23645', borderVisible: false, wickUpColor: '#089981', wickDownColor: '#f23645',
    });
    const volumeApi = instance.addSeries(HistogramSeries, { priceScaleId: 'volume', priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false });
    volumeApi.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    const lines = new Map<string, ChartSeries>();
    for (const spec of overlaySpec) {
      const color = spec.key === 'bollinger' ? '#59667c' : spec.color;
      const line = instance.addSeries(LineSeries, {
        color, lineWidth: spec.key === 'bollinger' ? 1 : 2, priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false, visible: overlays[spec.key], title: spec.label,
      });
      lines.set(spec.key, line);
      if (spec.key === 'bollinger') {
        lines.set('bollingerLower', instance.addSeries(LineSeries, {
          color: '#59667c', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false,
          crosshairMarkerVisible: false, visible: overlays.bollinger, title: 'Bollinger Lower',
        }));
      }
    }
    const zonesPrimitive = new SupplyDemandZonesPrimitive();
    candlesApi.attachPrimitive(zonesPrimitive);
    zonesPlugin.current = zonesPrimitive;

    chart.current = instance;
    candleSeries.current = candlesApi;
    volumeSeries.current = volumeApi;
    markerPlugin.current = createSeriesMarkers(candlesApi, []);
    overlaySeries.current = lines;
    instance.subscribeCrosshairMove((param) => {
      if (!param.point || param.time === undefined || param.time === null) {
        setHover(null);
        setHoveredZone(null);
        return;
      }
      const time = Number(param.time) as UTCTimestamp;
      const candle = candlesRef.current.find((item) => epoch(item.timestamp) === time);
      if (!candle) {
        setHover(null);
        setHoveredZone(null);
        return;
      }
      setHover({ x: param.point.x, y: param.point.y, candle, signal: signalsRef.current.get(time) });

      if (showZonesRef.current && zonesRef.current.length > 0 && candleSeries.current && chart.current) {
        const mouseX = param.point.x;
        const mouseY = param.point.y;
        const timeScale = chart.current.timeScale();
        const width = timeScale.width();

        let matched: Zone | null = null;
        for (const z of zonesRef.current) {
          const yUpper = candleSeries.current.priceToCoordinate(z.upper);
          const yLower = candleSeries.current.priceToCoordinate(z.lower);
          if (yUpper === null && yLower === null) continue;
          const top = Math.min(yUpper !== null ? Number(yUpper) : 0, yLower !== null ? Number(yLower) : 560);
          const bottom = Math.max(yUpper !== null ? Number(yUpper) : 0, yLower !== null ? Number(yLower) : 560);

          const fTime = z.formation_timestamp ? epoch(z.formation_timestamp) : null;
          let leftX = 0;
          if (fTime !== null && fTime > 0) {
            const visibleRange = timeScale.getVisibleRange();
            if (visibleRange && fTime > Number(visibleRange.to)) continue;

            let cX: number | null = null;
            const directCoord = timeScale.timeToCoordinate(fTime);
            if (directCoord !== null) {
              cX = Number(directCoord);
            } else {
              const barIdx = timeScale.timeToIndex(fTime, true);
              if (barIdx !== null) {
                const logicalCoord = timeScale.logicalToCoordinate(barIdx as unknown as import('lightweight-charts').Logical);
                if (logicalCoord !== null) cX = Number(logicalCoord);
              }
            }
            if (cX !== null) {
              leftX = Math.max(0, cX);
            } else {
              leftX = 0;
            }
          }

          let rightX = width;
          if (candlesRef.current.length > 0) {
            const lastIdx = candlesRef.current.length - 1;
            const coord = timeScale.logicalToCoordinate(lastIdx as unknown as import('lightweight-charts').Logical);
            if (coord !== null) {
              rightX = Math.min(width, Number(coord) + 4);
            }
          }
          if (z.status === 'INVALIDATED' && z.invalidation_timestamp) {
            const invTime = epoch(z.invalidation_timestamp);
            let invCoord: number | null = null;
            const direct = timeScale.timeToCoordinate(invTime);
            if (direct !== null) {
              invCoord = Number(direct);
            } else {
              const bIdx = timeScale.timeToIndex(invTime, true);
              if (bIdx !== null) {
                const lCoord = timeScale.logicalToCoordinate(bIdx as unknown as import('lightweight-charts').Logical);
                if (lCoord !== null) invCoord = Number(lCoord);
              }
            }
            if (invCoord !== null) {
              rightX = Math.max(leftX, Math.min(rightX, invCoord));
            }
          }

          if (mouseX >= leftX && mouseX <= rightX && mouseY >= top && mouseY <= bottom) {
            matched = z;
            break;
          }
        }
        setHoveredZone(matched);
      } else {
        setHoveredZone(null);
      }
    });
    const publishVisibleRightCandle = () => {
      const data = candlesRef.current;
      const range = instance.timeScale().getVisibleLogicalRange();
      if (!range || !data.length) return;
      const index = Math.min(data.length - 1, Math.floor(range.to));
      if (index >= 0) onVisibleRightCandleRef.current?.(data[index]);
    };
    instance.timeScale().subscribeVisibleLogicalRangeChange(publishVisibleRightCandle);
    const observer = new ResizeObserver(() => instance.applyOptions({ width: Math.max(1, node.clientWidth), height: Math.max(360, node.clientHeight) }));
    observer.observe(node);
    instance.applyOptions({ width: Math.max(1, node.clientWidth), height: Math.max(360, node.clientHeight) });
    return () => {
      observer.disconnect();
      instance.timeScale().unsubscribeVisibleLogicalRangeChange(publishVisibleRightCandle);
      markerPlugin.current?.detach();
      if (zonesPlugin.current && candleSeries.current) {
        candleSeries.current.detachPrimitive(zonesPlugin.current);
        zonesPlugin.current = null;
      }
      instance.remove();
      chart.current = null;
      candleSeries.current = null;
      volumeSeries.current = null;
      overlaySeries.current.clear();
    };
  // Create chart once. Data, overlays and chart controls update through refs below.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const calculated = useMemo(() => rollingLines(candles), [candles]);
  useEffect(() => {
    const instance = chart.current;
    const candleApi = candleSeries.current;
    const volumeApi = volumeSeries.current;
    if (!instance || !candleApi || !volumeApi) return;
    const displayCandles = formingCandle
      ? [...candles.filter((c) => c.timestamp !== formingCandle.timestamp), formingCandle].sort((a, b) => epoch(a.timestamp) - epoch(b.timestamp))
      : candles;
    candlesRef.current = displayCandles;
    const points = displayCandles.map((c) => ({ time: epoch(c.timestamp), open: c.open, high: c.high, low: c.low, close: c.close }));
    if (!plottedCandlesRef.current.length) {
      candleApi.setData(points);
      volumeApi.setData(displayCandles.map((c) => ({ time: epoch(c.timestamp), value: c.volume, color: c.close >= c.open ? '#08998177' : '#f2364577' })));
    } else {
      const previous = new Map(plottedCandlesRef.current.map((c) => [epoch(c.timestamp), c]));
      for (const c of displayCandles) {
        const t = epoch(c.timestamp); const old = previous.get(t);
        if (old && old.open === c.open && old.high === c.high && old.low === c.low && old.close === c.close && old.volume === c.volume) continue;
        const historical = Boolean(old) && t !== epoch(plottedCandlesRef.current.at(-1)!.timestamp);
        candleApi.update({ time: t, open: c.open, high: c.high, low: c.low, close: c.close }, historical);
        volumeApi.update({ time: t, value: c.volume, color: c.close >= c.open ? '#08998177' : '#f2364577' }, historical);
      }
    }
    plottedCandlesRef.current = displayCandles;
    for (const spec of overlaySpec) {
      const series = overlaySeries.current.get(spec.key);
      if (!series) continue;
      const group = calculated[spec.key];
      const active = spec.key === 'bollinger' ? overlays.bollinger : overlays[spec.key];
      series.applyOptions({ visible: active });
      if (active && group[0]) series.setData(group[0]);
      else series.setData([]);
      if (spec.key === 'bollinger') {
        const lower = overlaySeries.current.get('bollingerLower');
        lower?.applyOptions({ visible: active });
        if (active && group[1]) lower?.setData(group[1]);
        else lower?.setData([]);
      }
    }
    const markers: SeriesMarker<Time>[] = showMarkers ? signals.filter((s) => ['BUY', 'SELL'].includes(s.classification)).flatMap((s) => {
      const ts = epoch(s.candle_timestamp);
      if (!candles.some((point) => epoch(point.timestamp) === ts)) return [];
      const buy = s.classification === 'BUY';
      return [{ time: ts, position: buy ? 'belowBar' as const : 'aboveBar' as const, color: buy ? '#089981' : '#f23645', shape: buy ? 'arrowUp' as const : 'arrowDown' as const, text: buy ? 'BUY' : 'SELL', size: 1 }];
    }) : [];
    if (formingCandle && !candles.some((c) => c.timestamp === formingCandle.timestamp)) {
      markers.push({ time: epoch(formingCandle.timestamp), position: 'aboveBar', color: '#ff9800', shape: 'circle', text: 'FORMING', size: 1 });
    }
    markerPlugin.current?.setMarkers(markers);
    for (const line of priceLines.current) candleApi.removePriceLine(line);
    priceLines.current = [];

    // Supply & Demand Zones: Rectangular Bands (Finite boxes, no infinite price lines across chart)
    const supplies = (payload?.structure?.resistance_zones || payload?.structure?.supply_zones || []).slice(0, 5);
    const demands = (payload?.structure?.support_zones || payload?.structure?.demand_zones || []).slice(0, 5);
    const rawZones = showZones ? [...supplies, ...demands] : [];
    zonesRef.current = rawZones;
    const latestCandleTime = displayCandles.length ? epoch(displayCandles[displayCandles.length - 1].timestamp) : null;
    const latestCandleIndex = displayCandles.length > 0 ? displayCandles.length - 1 : null;
    zonesPlugin.current?.update(rawZones, latestCandleTime, latestCandleIndex);
    const risk = payload?.risk_plan;
    if (showRisk && risk?.valid) {
      const levels = [
        ['TRIGGER', risk.trigger_price, '#8290a7'], ['STOP', risk.stop_loss, '#f23645'],
        ['TARGET 1', risk.target_1, '#20a28a'], ['TARGET 2', risk.target_2, '#089981'],
      ] as const;
      levels.forEach(([title, value, color]) => {
        if (value != null) priceLines.current.push(candleApi.createPriceLine({ price: value, color, lineWidth: 1, lineStyle: title === 'TRIGGER' ? LineStyle.Dotted : LineStyle.Dashed, axisLabelVisible: false, title }));
      });
    }
    const addContextLine = (enabled: boolean, value: unknown, title: string, color: string, style: LineStyle) => {
      if (enabled && typeof value === 'number' && Number.isFinite(value)) {
        priceLines.current.push(candleApi.createPriceLine({ price: value, color, lineWidth: 1, lineStyle: style, axisLabelVisible: false, title }));
      }
    };
    const orb = intelligence?.opening_range;
    addContextLine(showOrbLevels && orb?.status === 'READY', orb?.high, 'ORB HIGH', '#ffb84d', LineStyle.Dashed);
    addContextLine(showOrbLevels && orb?.status === 'READY', orb?.low, 'ORB LOW', '#ffb84d', LineStyle.Dashed);
    addContextLine(showOrbLevels && orb?.status === 'READY', orb?.midpoint, 'ORB MID', '#a88343', LineStyle.Dotted);
    const prior = intelligence?.previous_day_levels;
    addContextLine(showPreviousDayLevels && prior?.status === 'READY', prior?.high, 'PREV HIGH', '#8d79d6', LineStyle.Dashed);
    addContextLine(showPreviousDayLevels && prior?.status === 'READY', prior?.low, 'PREV LOW', '#8d79d6', LineStyle.Dashed);
    addContextLine(showPreviousClose && prior?.status === 'READY', prior?.close, 'PREV CLOSE', '#64a8e4', LineStyle.Dotted);
    const signature = points.length ? `${points[0].time}:${points.at(-1)?.time}:${points.length}` : '';
    if (signature && signature !== rangeSignature.current) {
      if (rangeSignature.current && autoScroll) instance.timeScale().scrollToRealTime();
      else if (!rangeSignature.current) instance.timeScale().fitContent();
      rangeSignature.current = signature;
    }
    const range = instance.timeScale().getVisibleLogicalRange();
    if (range && candles.length) {
      const index = Math.min(candles.length - 1, Math.floor(range.to));
      if (index >= 0) onVisibleRightCandleRef.current?.(candles[index]);
    }
  }, [candles, formingCandle, calculated, signals, payload, intelligence, overlays, showMarkers, showRisk, showZones, showOrbLevels, showPreviousDayLevels, showPreviousClose, autoScroll]);

  useEffect(() => {
    if (!selectedTimestamp || !chart.current) return;
    const index = candles.findIndex((item) => epoch(item.timestamp) === epoch(selectedTimestamp));
    if (index >= 0) chart.current.timeScale().setVisibleLogicalRange({ from: Math.max(0, index - 35), to: Math.min(candles.length - 1, index + 35) });
  }, [selectedTimestamp, candles]);

  const lineValues = useMemo(() => {
    const ema9 = new Map((calculated.ema9[0] || []).map((line) => [line.time, line.value]));
    const ema20 = new Map((calculated.ema20[0] || []).map((line) => [line.time, line.value]));
    return new Map(candles.map((item) => [epoch(item.timestamp), { ema9: ema9.get(epoch(item.timestamp)), ema20: ema20.get(epoch(item.timestamp)) }]));
  }, [calculated, candles]);
  const selectedInRange = selectedTimestamp ? candles.some((item) => epoch(item.timestamp) === epoch(selectedTimestamp)) : true;

  return <div className="market-chart-card">
    <div className="chart-toolbar">
      <div className="chart-tool-group"><span className="toolbar-label">OVERLAYS</span>{overlaySpec.map((item) => <label key={item.key} className={`overlay-toggle ${overlays[item.key] ? 'selected' : ''}`} title={`${item.label} · computed from completed actual candles`}><input type="checkbox" checked={overlays[item.key]} onChange={(event) => setOverlays((old) => ({ ...old, [item.key]: event.target.checked }))} /><i style={{ background: item.color }} />{item.label}</label>)}</div>
      <div className="chart-tool-group chart-actions">
        <label className="compact-check"><input type="checkbox" checked={showVolume} onChange={(event) => { setShowVolume(event.target.checked); volumeSeries.current?.applyOptions({ visible: event.target.checked }); }} />Volume</label>
        <label className="compact-check"><input type="checkbox" checked={showZones} onChange={(event) => setShowZones(event.target.checked)} />Supply/Demand zones</label>
        <label className="compact-check"><input type="checkbox" checked={showRisk} onChange={(event) => setShowRisk(event.target.checked)} />Risk levels</label>
        <label className="compact-check"><input type="checkbox" checked={showOrbLevels} onChange={(event) => setShowOrbLevels(event.target.checked)} />Opening range</label>
        <label className="compact-check"><input type="checkbox" checked={showPreviousDayLevels} onChange={(event) => setShowPreviousDayLevels(event.target.checked)} />Prev high/low</label>
        <label className="compact-check"><input type="checkbox" checked={showPreviousClose} onChange={(event) => setShowPreviousClose(event.target.checked)} />Prev close</label>
        <label className="compact-check"><input type="checkbox" checked={showMarkers} onChange={(event) => setShowMarkers(event.target.checked)} />BUY/SELL</label>
        <label className="compact-check"><input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.target.checked)} />Auto-scroll</label>
        <button className="terminal-icon-button" onClick={() => chart.current?.timeScale().fitContent()} title="Fit all displayed candles">Reset</button>
        <button className="terminal-icon-button" onClick={() => { if (document.fullscreenElement) void document.exitFullscreen(); else void host.current?.parentElement?.requestFullscreen(); }} title="Toggle chart fullscreen">Fullscreen</button>
      </div>
    </div>
    <div className="chart-frame">
      <div ref={host} className="trading-chart" aria-label="AMZN actual 5-minute candlestick chart" />
      {hover && <div className="market-crosshair-tooltip" style={{ left: Math.min(Math.max(hover.x + 15, 10), Math.max(10, (host.current?.clientWidth || 700) - 250)), top: Math.min(Math.max(hover.y + 12, 10), 360) }}>
        <b>{hover.signal ? <span className={`decision-text ${hover.signal.classification === 'BUY' ? 'positive' : hover.signal.classification === 'SELL' ? 'negative' : ''}`}>{hover.signal.classification}</span> : 'AMZN · ACTUAL 5M'}</b>
        <span>{formatTime(hover.candle.timestamp, true)}</span>
        <div className="ohlc-grid"><span>O {price(hover.candle.open)}</span><span>H {price(hover.candle.high)}</span><span>L {price(hover.candle.low)}</span><span>C {price(hover.candle.close)}</span><span>V {hover.candle.volume.toLocaleString()}</span></div>
        {hover.signal && <>
          <span>Decision score {Number(hover.signal.final_score ?? 0).toFixed(3)} · heuristic, not probability</span>
          <span>Raw ML P(UP) {hover.signal.payload.ml_evidence?.raw_probability_up == null ? '—' : `${(hover.signal.payload.ml_evidence.raw_probability_up * 100).toFixed(1)}% · uncalibrated`}</span>
          <span>Status {hover.signal.status} · hypothetical only</span>
        </>}
        {lineValues.get(epoch(hover.candle.timestamp))?.ema9 != null && <span>EMA 9 {price(lineValues.get(epoch(hover.candle.timestamp))?.ema9)} · EMA 20 {price(lineValues.get(epoch(hover.candle.timestamp))?.ema20)}</span>}
        {hoveredZone && (
          <div className="zone-tooltip-section">
            <div className="zone-tooltip-header">
              <span className={`zone-tag ${(hoveredZone.zone_type || (hoveredZone.type === 'RESISTANCE' ? 'SUPPLY' : 'DEMAND')).toLowerCase()}`}>
                {hoveredZone.zone_type || (hoveredZone.type === 'RESISTANCE' ? 'SUPPLY' : 'DEMAND')} ZONE
              </span>
              <span className={`zone-badge ${(hoveredZone.status || 'ACTIVE').toLowerCase()}`}>
                {hoveredZone.status || 'ACTIVE'}
              </span>
            </div>
            <div className="zone-detail-row">
              <span className="zone-k">Range:</span>
              <span className="zone-v">${hoveredZone.lower.toFixed(2)} - ${hoveredZone.upper.toFixed(2)}</span>
            </div>
            <div className="zone-detail-row">
              <span className="zone-k">Formed:</span>
              <span className="zone-v">{formatTime(hoveredZone.formation_timestamp || hover.candle.timestamp, true)}</span>
            </div>
            <div className="zone-detail-row">
              <span className="zone-k">Touches:</span>
              <span className="zone-v">{hoveredZone.touches ?? 1}</span>
            </div>
            {hoveredZone.distance_from_price != null && (
              <div className="zone-detail-row">
                <span className="zone-k">Distance:</span>
                <span className="zone-v">
                  ${Math.abs(hoveredZone.distance_from_price).toFixed(2)}
                  {hoveredZone.distance_atr != null ? ` (${Math.abs(hoveredZone.distance_atr).toFixed(1)} ATR)` : ''}
                </span>
              </div>
            )}
            {hoveredZone.strength_score != null && (
              <div className="zone-detail-row">
                <span className="zone-k">Quality:</span>
                <span className="zone-v">{hoveredZone.strength_score.toFixed(0)}/100</span>
              </div>
            )}
          </div>
        )}
      </div>}
      {!candles.length && <div className="chart-empty-overlay">Waiting for actual processed AMZN candles…</div>}
      {candles.length > 0 && !selectedInRange && <div className="chart-empty-overlay selected-outside-range">Selected signal is outside the latest 1,000 plotted candles.</div>}
    </div>
    <div className="chart-legend-row"><span className="legend-chip"><i className="legend-bull" /> Actual up candle</span><span className="legend-chip"><i className="legend-bear" /> Actual down candle</span><span className="legend-chip"><i style={{ display: 'inline-block', width: 8, height: 8, background: '#f2364533', border: '1px solid #f23645' }} /> Supply zone</span><span className="legend-chip"><i style={{ display: 'inline-block', width: 8, height: 8, background: '#08998133', border: '1px solid #089981' }} /> Demand zone</span><span className="legend-chip"><i className="legend-volume" /> Actual volume</span><span className="legend-chip">Indicators & zones from actual completed 5m candles</span></div>
  </div>;
}
