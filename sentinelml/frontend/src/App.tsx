import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, BrainCircuit,
  Check, CheckCircle2, ChevronRight, CircleHelp, Database, Gauge,
  Layers3, LayoutDashboard, Menu, Moon, Network, RefreshCw, Search,
  ShieldCheck, Sun, TerminalSquare, TriangleAlert, Zap,
} from 'lucide-react';
import {
  Button, InlineNotification, NumberInput, Tag, Theme, Tile,
} from '@carbon/react';
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

type Page = 'overview' | 'predictions' | 'drift' | 'incidents' | 'models';
type ModelInfo = {
  model_version: string; created_at?: string; algorithm?: string; task?: string; features?: string[];
  training_rows?: number; train_rows?: number; test_rows?: number;
  metrics?: Record<string, number>; data_validation?: { valid: boolean; rows?: number };
  dataset?: { type?: string; generator?: string };
};
type Health = { status: string; model_loaded: boolean };
type Metrics = { requests: number; prediction_requests: number; errors: number; average_latency_ms: number; total_latency_ms: number };
type Feature = { psi: number; level: 'low' | 'moderate' | 'significant' };
type DriftReport = { rows: number; features: Record<string, Feature>; summary: { significant_features: string[]; moderate_features: string[]; status: string; recommendation: string }; method: string; caveat: string };
type PredictionResult = { model_version: string; predictions: { prediction: number; probability_class_1: number }[] };
type IncidentResult = Record<string, unknown>;

const FEATURES = Array.from({ length: 10 }, (_, i) => `feature_${i}`);
const apiBase = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
const navItems: { id: Page; label: string; hint: string; icon: typeof LayoutDashboard }[] = [
  { id: 'overview', label: 'Overview', hint: 'System status', icon: LayoutDashboard },
  { id: 'predictions', label: 'Predictions', hint: 'Run inference', icon: BrainCircuit },
  { id: 'drift', label: 'Drift monitor', hint: 'Data quality', icon: Activity },
  { id: 'incidents', label: 'AI analyst', hint: 'Investigate signals', icon: Zap },
  { id: 'models', label: 'Model details', hint: 'Active version', icon: Layers3 },
];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as T;
}

function toInstances(values: number[]): { instances: Record<string, number>[] } {
  return { instances: [Object.fromEntries(FEATURES.map((feature, i) => [feature, values[i] ?? 0]))] };
}

function fmt(value: number | undefined, digits = 3) {
  if (value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}
function metricLabel(key: string) { return key.replaceAll('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase()); }
function dateLabel(value?: string) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

export default function App() {
  const [page, setPage] = useState<Page>('overview');
  const [dark, setDark] = useState(() => localStorage.getItem('sentinelml-theme') !== 'light');
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [apiError, setApiError] = useState('');
  const [featureValues, setFeatureValues] = useState<number[]>([0.25, -0.4, 1.1, 0.7, -0.2, 0.9, 0.1, -0.6, 0.3, 1.2]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState('');
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [checkingDrift, setCheckingDrift] = useState(false);
  const [driftError, setDriftError] = useState('');
  const [incident, setIncident] = useState<IncidentResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [incidentError, setIncidentError] = useState('');
  const [events, setEvents] = useState<{ id: number; time: string; title: string; detail: string; kind: 'success' | 'info' | 'warning' }[]>([]);

  const addEvent = useCallback((title: string, detail: string, kind: 'success' | 'info' | 'warning' = 'info') => {
    setEvents((current) => [{ id: Date.now(), time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), title, detail, kind }, ...current].slice(0, 5));
  }, []);

  const refresh = useCallback(async (quiet = false) => {
    if (quiet) setRefreshing(true); else setLoading(true);
    setApiError('');
    try {
      const [h, m, met] = await Promise.all([
        api<Health>('/health'), api<ModelInfo>('/model-info'), api<Metrics>('/metrics'),
      ]);
      setHealth(h); setModel(m); setMetrics(met);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Unable to reach SentinelML API.');
      setHealth(null); setModel(null); setMetrics(null);
    } finally { setLoading(false); setRefreshing(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { localStorage.setItem('sentinelml-theme', dark ? 'dark' : 'light'); }, [dark]);
  useEffect(() => {
    if (!health?.model_loaded) return;
    const id = window.setInterval(() => { void refresh(true); }, 30000);
    return () => window.clearInterval(id);
  }, [health?.model_loaded, refresh]);

  const modelMetrics = useMemo(() => Object.entries(model?.metrics || {}), [model]);
  const qualitySeries = useMemo(() => modelMetrics.map(([name, value]) => ({ name: metricLabel(name), value: Number((value * 100).toFixed(1)) })), [modelMetrics]);

  async function runPrediction() {
    setPredicting(true); setPredictionError(''); setPrediction(null);
    try {
      const result = await api<PredictionResult>('/predict', { method: 'POST', body: JSON.stringify(toInstances(featureValues)) });
      setPrediction(result); addEvent('Prediction completed', `One record evaluated by ${result.model_version}.`, 'success');
      void refresh(true);
    } catch (error) { setPredictionError(error instanceof Error ? error.message : 'Prediction failed.'); }
    finally { setPredicting(false); }
  }

  function createBatch() {
    // Deterministic perturbations around the current values, so repeated runs are reproducible.
    return { instances: Array.from({ length: 30 }, (_, row) => Object.fromEntries(FEATURES.map((feature, i) => {
      const wave = Math.sin((row + 1) * (i + 2) * 0.37) * 0.35;
      return [feature, Number((featureValues[i] + wave).toFixed(5))];
    }))) };
  }
  async function runDrift() {
    setCheckingDrift(true); setDriftError(''); setDrift(null);
    try {
      const report = await api<DriftReport>('/monitor/drift', { method: 'POST', body: JSON.stringify(createBatch()) });
      setDrift(report);
      const kind = report.summary.status === 'drift_detected' ? 'warning' : report.summary.status === 'review' ? 'info' : 'success';
      addEvent('Drift check completed', `${report.rows} records checked · ${report.summary.significant_features.length} significant features.`, kind);
      void refresh(true);
    } catch (error) { setDriftError(error instanceof Error ? error.message : 'Drift check failed.'); }
    finally { setCheckingDrift(false); }
  }
  async function runIncident() {
    setAnalyzing(true); setIncidentError(''); setIncident(null);
    try {
      const result = await api<IncidentResult>('/ai/analyze', { method: 'POST', body: JSON.stringify(createBatch()) });
      setIncident(result); addEvent('AI analysis completed', 'Evidence-based advisory returned by the configured provider.', 'success');
    } catch (error) { setIncidentError(error instanceof Error ? error.message : 'AI analysis failed.'); }
    finally { setAnalyzing(false); }
  }

  const currentTitle = navItems.find((item) => item.id === page)?.label || 'Overview';
  return (
    <Theme theme={dark ? 'g100' : 'white'}>
      <div className={`app-shell ${dark ? 'theme-dark' : 'theme-light'}`}>
        <header className="topbar">
          <div className="brand-lockup">
            <button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileNav((v) => !v)}><Menu size={20} /></button>
            <div className="brand-mark"><ShieldCheck size={19} strokeWidth={2.2} /></div>
            <div className="brand-text"><span className="brand-name">sentinel<span>ml</span></span><span className="brand-divider" /><span className="brand-product">MODEL OPERATIONS</span></div>
          </div>
          <div className="topbar-right">
            <div className="environment"><span className={`status-dot ${health?.model_loaded ? 'is-live' : 'is-down'}`} />{health?.model_loaded ? 'Local environment' : 'API unavailable'}</div>
            <div className="topbar-separator" />
            <button className="icon-button theme-switch" aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} title={dark ? 'Switch to light theme' : 'Switch to dark theme'} onClick={() => setDark((v) => !v)}>{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
            <button className="profile-button" aria-label="SentinelML workspace">S</button>
          </div>
        </header>

        <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
          <div className="workspace-label">WORKSPACE</div>
          <div className="workspace-switcher"><div className="workspace-icon"><Network size={17} /></div><div className="workspace-meta"><strong>Local deployment</strong><span>SentinelML · dev</span></div><ChevronRight size={16} className="workspace-chevron" /></div>
          <nav aria-label="Primary navigation">
            <div className="nav-section-label">OPERATIONS</div>
            {navItems.map(({ id, label, hint, icon: Icon }) => (
              <button key={id} className={`nav-item ${page === id ? 'nav-item-active' : ''}`} onClick={() => { setPage(id); setMobileNav(false); }} aria-current={page === id ? 'page' : undefined}>
                <Icon size={18} strokeWidth={1.8} /><span className="nav-copy"><span>{label}</span><small>{hint}</small></span>{page === id && <span className="nav-active-bar" />}
              </button>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="sidebar-health"><span className={`status-dot ${health?.model_loaded ? 'is-live' : 'is-down'}`} /><div><strong>{health?.model_loaded ? 'Model service ready' : 'Service not ready'}</strong><small>{health?.model_loaded ? 'Health check passing' : 'Check API connection'}</small></div></div>
            <div className="sidebar-footer"><span>SentinelML</span><span>v1.0.0</span></div>
          </div>
        </aside>
        {mobileNav && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}

        <main className="main-content">
          <div className="page-topline"><div className="breadcrumbs"><span>SentinelML</span><ChevronRight size={13} /><strong>{currentTitle}</strong></div><div className="page-actions"><span className="last-updated">{loading ? 'Connecting…' : `Updated ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`}</span><Button kind="ghost" size="sm" renderIcon={RefreshCw} hasIconOnly iconDescription="Refresh status" className={refreshing ? 'refreshing' : ''} onClick={() => void refresh(true)} /></div></div>
          {apiError && <InlineNotification kind="error" lowContrast title="Backend connection failed" subtitle={`${apiError}. Start the FastAPI server on port 8000, then refresh.`} hideCloseButton className="global-notification" />}
          {page === 'overview' && <Overview model={model} health={health} metrics={metrics} loading={loading} qualitySeries={qualitySeries} events={events} onNavigate={setPage} onRefresh={() => void refresh(true)} refreshing={refreshing} />}
          {page === 'predictions' && <Predictions values={featureValues} setValues={setFeatureValues} result={prediction} error={predictionError} busy={predicting} onRun={runPrediction} onReset={() => { setFeatureValues(Array(10).fill(0)); setPrediction(null); setPredictionError(''); }} />}
          {page === 'drift' && <DriftPage report={drift} error={driftError} busy={checkingDrift} onRun={runDrift} />}
          {page === 'incidents' && <IncidentPage report={incident} error={incidentError} busy={analyzing} onRun={runIncident} onDrift={() => { setPage('drift'); void runDrift(); }} />}
          {page === 'models' && <ModelPage model={model} health={health} loading={loading} metrics={metrics} onRefresh={() => void refresh(true)} />}
          <footer className="page-footer"><span><span className="footer-status-dot" /> Operational data from your local SentinelML API</span><span>Built for transparent ML operations <span className="footer-divider">/</span> Not a production certification</span></footer>
        </main>
      </div>
    </Theme>
  );
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>;
}
function SectionHeading({ title, caption, action }: { title: string; caption?: string; action?: React.ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{caption && <p>{caption}</p>}</div>{action}</div>;
}
function MetricCard({ label, value, foot, icon: Icon, accent = 'blue', trend }: { label: string; value: string; foot: string; icon: typeof Activity; accent?: string; trend?: 'up' | 'down' }) {
  return <Tile className={`metric-card metric-${accent}`}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-foot">{trend === 'up' ? <ArrowUpRight size={13} /> : trend === 'down' ? <ArrowDownRight size={13} /> : <span className="metric-foot-dot" />}{foot}</div></Tile>;
}
function Overview({ model, health, metrics, loading, qualitySeries, events, onNavigate, onRefresh, refreshing }: { model: ModelInfo | null; health: Health | null; metrics: Metrics | null; loading: boolean; qualitySeries: { name: string; value: number }[]; events: { id: number; time: string; title: string; detail: string; kind: string }[]; onNavigate: (page: Page) => void; onRefresh: () => void; refreshing: boolean }) {
  const roc = model?.metrics?.roc_auc;
  return <>
    <PageHeading eyebrow="OPERATIONS / OVERVIEW" title="Model operations" description="A clear view of model health, serving activity, and data quality signals." action={<Button kind="secondary" size="md" renderIcon={TerminalSquare} onClick={() => onNavigate('models')}>Inspect model</Button>} />
    <div className="overview-status-row"><div className="status-summary"><span className={`large-status-mark ${health?.model_loaded ? 'mark-good' : 'mark-bad'}`}>{health?.model_loaded ? <Check size={17} /> : <TriangleAlert size={17} />}</span><div><strong>{health?.model_loaded ? 'Model service is ready' : 'Waiting for model service'}</strong><span>{model?.model_version || 'No active model metadata available'}</span></div><Tag type={health?.model_loaded ? 'green' : 'red'}>{health?.model_loaded ? 'READY' : 'OFFLINE'}</Tag></div><div className="status-summary-right"><span className="live-pulse" /> Live API status <span className="status-divider">·</span> Auto-refresh 30s</div></div>
    <div className="metric-grid">
      <MetricCard label="ROC–AUC" value={loading ? '—' : fmt(roc, 3)} foot="Held-out evaluation" icon={Gauge} accent="blue" />
      <MetricCard label="Predictions served" value={loading ? '—' : (metrics?.prediction_requests ?? 0).toLocaleString()} foot="Since API process started" icon={BrainCircuit} accent="purple" />
      <MetricCard label="Avg. latency" value={loading ? '—' : `${fmt(metrics?.average_latency_ms, 1)} ms`} foot="Across API requests" icon={Activity} accent="teal" />
      <MetricCard label="API errors" value={loading ? '—' : (metrics?.errors ?? 0).toLocaleString()} foot="Observed in this process" icon={ShieldCheck} accent={(metrics?.errors ?? 0) > 0 ? 'amber' : 'green'} />
    </div>
    <div className="content-grid overview-grid">
      <Tile className="panel quality-panel"><SectionHeading title="Model quality" caption="Evaluation metrics from the active model artifact" action={<Tag type="blue">HELD-OUT SET</Tag>} />
        {qualitySeries.length ? <><div className="quality-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={qualitySeries} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="var(--border-subtle)" /><XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} /><Tooltip contentStyle={{ background: 'var(--layer-floating)', border: '1px solid var(--border-strong)', borderRadius: 2, color: 'var(--text-primary)', fontSize: 12 }} formatter={(value) => [`${value}%`, 'Score']} /><Bar dataKey="value" fill="#4589ff" radius={[2, 2, 0, 0]} maxBarSize={36} /></BarChart></ResponsiveContainer></div><div className="metric-mini-grid">{modelMetricsFrom(model).map(([key, value]) => <div className="metric-mini" key={key}><span>{metricLabel(key)}</span><strong>{fmt(value, 3)}</strong><div className="mini-track"><span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>)}</div></> : <EmptyInline title="Model metrics unavailable" detail="Start the backend and load model metadata to see evaluation metrics." />}
        <div className="panel-bottom-note"><CircleHelp size={14} /><span>Scores describe a synthetic evaluation dataset; validate on representative data before real use.</span></div>
      </Tile>
      <div className="right-stack">
        <Tile className="panel model-summary-panel"><SectionHeading title="Active model" caption="Currently loaded artifact" /><div className="model-name-row"><div className="model-symbol"><Layers3 size={19} /></div><div><strong>{model?.algorithm || 'Model unavailable'}</strong><span>{model?.model_version || 'No active version'}</span></div></div><div className="detail-pairs"><div><span>Task</span><strong>{model?.task?.replaceAll('_', ' ') || '—'}</strong></div><div><span>Training rows</span><strong>{model?.training_rows?.toLocaleString() || '—'}</strong></div><div><span>Features</span><strong>{model?.features?.length ?? 10}</strong></div><div><span>Created</span><strong>{dateLabel(model?.created_at)}</strong></div></div><Button kind="ghost" size="sm" className="panel-link-button" renderIcon={ArrowRight} onClick={() => onNavigate('models')}>View model details</Button></Tile>
        <Tile className="panel quick-actions-panel"><SectionHeading title="Quick actions" caption="Run an operation" /><button className="quick-action" onClick={() => onNavigate('predictions')}><span className="quick-action-icon"><BrainCircuit size={17} /></span><span><strong>Run a prediction</strong><small>Evaluate a single feature vector</small></span><ChevronRight size={16} /></button><button className="quick-action" onClick={() => onNavigate('drift')}><span className="quick-action-icon quick-action-icon-teal"><Activity size={17} /></span><span><strong>Check feature drift</strong><small>Compare a batch against baseline</small></span><ChevronRight size={16} /></button><button className="quick-action" onClick={() => onNavigate('incidents')}><span className="quick-action-icon quick-action-icon-purple"><Zap size={17} /></span><span><strong>Investigate with AI</strong><small>Generate an advisory from drift evidence</small></span><ChevronRight size={16} /></button></Tile>
      </div>
    </div>
    <div className="content-grid bottom-grid"><Tile className="panel activity-panel"><SectionHeading title="Session activity" caption="Actions performed in this browser session" action={<Button kind="ghost" size="sm" onClick={onRefresh} disabled={refreshing} renderIcon={RefreshCw}>Refresh data</Button>} />{events.length ? <div className="activity-list">{events.map((event) => <div className="activity-row" key={event.id}><span className={`activity-symbol activity-${event.kind}`}>{event.kind === 'success' ? <CheckCircle2 size={16} /> : event.kind === 'warning' ? <TriangleAlert size={16} /> : <Activity size={16} />}</span><div className="activity-copy"><strong>{event.title}</strong><span>{event.detail}</span></div><time>{event.time}</time></div>)}</div> : <EmptyInline title="No activity yet" detail="Run a prediction or drift check. Results will appear here for this session." />}</Tile><Tile className="panel data-note-panel"><div className="data-note-icon"><Database size={19} /></div><div className="eyebrow">DATA PROVENANCE</div><h3>{model?.dataset?.type === 'synthetic' ? 'Synthetic reference data' : model?.dataset?.type || 'Dataset metadata unavailable'}</h3><p>The active artifact records its training source and evaluation context. Current PSI checks are heuristic signals—not a direct measurement of model degradation.</p><Button kind="tertiary" size="sm" renderIcon={ArrowRight} onClick={() => onNavigate('drift')}>Understand drift</Button></Tile></div>
  </>;
}
function modelMetricsFrom(model: ModelInfo | null) { return Object.entries(model?.metrics || {}); }
function EmptyInline({ title, detail }: { title: string; detail: string }) { return <div className="empty-inline"><div className="empty-icon"><Search size={18} /></div><strong>{title}</strong><p>{detail}</p></div>; }

function FeatureEditor({ values, setValues }: { values: number[]; setValues: (values: number[]) => void }) {
  return <div className="feature-grid">{FEATURES.map((feature, i) => <div className="feature-input" key={feature}><label htmlFor={feature}>{feature.replace('_', ' ')}</label><NumberInput id={feature} label={feature} hideLabel min={-100000} max={100000} step={0.1} value={values[i]} onChange={(_, state) => { const next = [...values]; next[i] = Number(state.value ?? 0); setValues(next); }} size="md" /></div>)}</div>;
}
function Predictions({ values, setValues, result, error, busy, onRun, onReset }: { values: number[]; setValues: (values: number[]) => void; result: PredictionResult | null; error: string; busy: boolean; onRun: () => void; onReset: () => void }) {
  const prediction = result?.predictions[0];
  return <><PageHeading eyebrow="OPERATIONS / INFERENCE" title="Prediction playground" description="Send one feature vector to the active model and inspect its response. No input is saved by this interface." action={<Tag type="blue">LIVE INFERENCE</Tag>} />
    <div className="content-grid form-layout"><Tile className="panel form-panel"><SectionHeading title="Input features" caption="Enter values for all 10 model features" action={<Button kind="ghost" size="sm" onClick={onReset}>Reset values</Button>} /><div className="form-callout"><CircleHelp size={16} /><span>The demo model was trained on synthetic data. These outputs are for technical demonstration only.</span></div><FeatureEditor values={values} setValues={setValues} /><div className="form-footer"><span className="form-footnote">10 numeric fields <span>·</span> 1 record</span><Button kind="primary" onClick={onRun} disabled={busy} renderIcon={ArrowRight}>{busy ? 'Running inference…' : 'Run prediction'}</Button></div>{error && <InlineNotification kind="error" title="Prediction failed" subtitle={error} lowContrast className="result-notification" />}</Tile>
      <Tile className="panel result-panel"><SectionHeading title="Inference result" caption="Response from the API" />{busy ? <div className="result-loading"><div className="loading-orbit" /><strong>Evaluating input</strong><span>Waiting for model response</span></div> : prediction ? <div className="prediction-result"><div className="result-overline"><CheckCircle2 size={15} /> PREDICTION COMPLETE</div><div className="prediction-class"><span>Predicted class</span><strong>{prediction.prediction}</strong><Tag type={prediction.prediction === 1 ? 'blue' : 'cool-gray'}>CLASS {prediction.prediction}</Tag></div><div className="probability-block"><div className="probability-heading"><span>Probability of class 1</span><strong>{(prediction.probability_class_1 * 100).toFixed(2)}%</strong></div><div className="probability-track"><span style={{ width: `${prediction.probability_class_1 * 100}%` }} /></div><div className="probability-scale"><span>0%</span><span>100%</span></div></div><div className="result-meta"><span>Model version</span><code>{result.model_version}</code></div><div className="result-disclaimer"><TriangleAlert size={15} /><span>Model confidence is not a guarantee of correctness. Interpret predictions only in the context of validated data and a defined use case.</span></div></div> : <div className="result-empty"><div className="result-empty-mark"><BrainCircuit size={25} /></div><h3>Ready for inference</h3><p>Enter feature values and run a prediction. The model response will appear here.</p><div className="result-empty-meta"><span>Endpoint</span><code>POST /predict</code></div></div>}</Tile></div>
  </>;
}
function DriftPage({ report, error, busy, onRun }: { report: DriftReport | null; error: string; busy: boolean; onRun: () => void }) {
  const sorted = Object.entries(report?.features || {}).sort((a, b) => b[1].psi - a[1].psi);
  const significant = report?.summary.significant_features.length || 0;
  const moderate = report?.summary.moderate_features.length || 0;
  return <><PageHeading eyebrow="OPERATIONS / MONITORING" title="Drift monitor" description="Compare an incoming batch against the model's stored reference distribution." action={<Button kind="primary" renderIcon={RefreshCw} onClick={onRun} disabled={busy}>{busy ? 'Checking batch…' : 'Run drift check'}</Button>} />
    <div className="drift-context-strip"><div className="context-symbol"><Activity size={18} /></div><div><strong>Population Stability Index (PSI)</strong><span>Reference-quantile bins · 30-record deterministic demo batch</span></div><Tag type="cool-gray">HEURISTIC</Tag><div className="context-right"><CircleHelp size={15} /><span>Drift is not the same as model degradation.</span></div></div>
    {error && <InlineNotification kind="error" title="Drift check failed" subtitle={error} lowContrast className="result-notification" />}
    {report ? <><div className="drift-summary-grid"><Tile className="panel drift-summary-card"><span className="summary-label">Batch size</span><strong>{report.rows.toLocaleString()}</strong><span className="summary-foot">Records evaluated</span></Tile><Tile className="panel drift-summary-card"><span className="summary-label">Significant features</span><strong className={significant ? 'text-warning' : ''}>{significant}</strong><span className="summary-foot">PSI ≥ 0.20</span></Tile><Tile className="panel drift-summary-card"><span className="summary-label">Moderate features</span><strong>{moderate}</strong><span className="summary-foot">PSI 0.10–0.20</span></Tile><Tile className="panel drift-summary-card"><span className="summary-label">Overall assessment</span><Tag type={report.summary.status === 'drift_detected' ? 'red' : report.summary.status === 'review' ? 'warm-gray' : 'green'}>{report.summary.status.replaceAll('_', ' ').toUpperCase()}</Tag><span className="summary-foot">Based on current batch</span></Tile></div>
      <Tile className="panel drift-table-panel"><SectionHeading title="Feature-level drift" caption="Sorted by highest PSI score" action={<span className="table-count">{sorted.length} features</span>} /><div className="drift-table-wrap"><table className="data-table"><thead><tr><th>Feature</th><th>PSI score</th><th>Relative signal</th><th>Assessment</th></tr></thead><tbody>{sorted.map(([name, feature]) => <tr key={name}><td><span className="feature-name"><span className="feature-index">{name.split('_')[1]}</span>{name}</span></td><td className="psi-value">{feature.psi.toFixed(5)}</td><td><div className="psi-track"><span className={feature.level} style={{ width: `${Math.min(100, feature.psi / 0.4 * 100)}%` }} /></div></td><td><Tag type={feature.level === 'significant' ? 'red' : feature.level === 'moderate' ? 'warm-gray' : 'green'}>{feature.level}</Tag></td></tr>)}</tbody></table></div><div className="drift-method"><span><Database size={14} /> {report.method}</span><span>Thresholds: low &lt; 0.10 · moderate 0.10–0.20 · significant ≥ 0.20</span></div></Tile>
      <div className={`recommendation-card ${report.summary.status === 'drift_detected' ? 'recommendation-warning' : ''}`}><div className="recommendation-icon">{report.summary.status === 'drift_detected' ? <TriangleAlert size={18} /> : <ShieldCheck size={18} />}</div><div><strong>Recommended next step</strong><p>{report.summary.recommendation}</p><small>{report.caveat}</small></div></div>
    </> : <Tile className="panel drift-empty"><div className="drift-empty-art"><div className="drift-empty-ring ring-one" /><div className="drift-empty-ring ring-two" /><div className="drift-empty-core"><Activity size={26} /></div></div><div className="eyebrow">MONITORING WORKFLOW</div><h2>No drift report yet</h2><p>Run a check to score the current demo batch against the reference distribution stored with your active model.</p><Button kind="primary" renderIcon={ArrowRight} onClick={onRun} disabled={busy}>{busy ? 'Checking batch…' : 'Run first drift check'}</Button><span className="empty-note">The generated batch is deterministic and intended for local demonstration.</span></Tile>}
  </>;
}
function IncidentPage({ report, error, busy, onRun, onDrift }: { report: IncidentResult | null; error: string; busy: boolean; onRun: () => void; onDrift: () => void }) {
  return <><PageHeading eyebrow="OPERATIONS / INVESTIGATION" title="AI incident analyst" description="Ask an LLM to interpret a drift report and propose investigation steps. Recommendations remain advisory." action={<Tag type="purple">OPTIONAL PROVIDER</Tag>} />
    <div className="incident-guardrail"><div className="guardrail-icon"><ShieldCheck size={19} /></div><div><strong>Human-controlled by design</strong><p>The analyst receives model metadata and drift evidence. It cannot promote, retrain, or deploy a model. An OpenAI API key is required for live analysis.</p></div></div>
    <div className="content-grid incident-layout"><Tile className="panel incident-run-panel"><div className="incident-symbol"><Zap size={22} /></div><div className="eyebrow">EVIDENCE-BASED ANALYSIS</div><h2>Investigate a signal</h2><p>SentinelML will calculate drift on a 30-record demo batch and pass the resulting evidence to the configured AI provider.</p><div className="incident-steps"><div><span>01</span><div><strong>Build evidence</strong><small>Feature-level PSI report</small></div></div><div><span>02</span><div><strong>Analyze context</strong><small>Active model metadata + drift report</small></div></div><div><span>03</span><div><strong>Review recommendations</strong><small>Human review required</small></div></div></div><Button kind="primary" renderIcon={ArrowRight} onClick={onRun} disabled={busy}>{busy ? 'Analyzing evidence…' : 'Analyze with AI'}</Button><Button kind="ghost" renderIcon={Activity} onClick={onDrift} className="secondary-full-button">Run drift check only</Button>{error && <InlineNotification kind="error" title="Analysis unavailable" subtitle={error} lowContrast className="result-notification" />}</Tile>
      <Tile className="panel incident-result-panel"><SectionHeading title="Analysis output" caption="Provider response, shown as returned" />{busy ? <div className="result-loading"><div className="loading-orbit" /><strong>Reviewing evidence</strong><span>Waiting for configured AI provider</span></div> : report ? <pre className="json-output">{JSON.stringify(report, null, 2)}</pre> : <div className="result-empty"><div className="result-empty-mark"><BrainCircuit size={25} /></div><h3>Awaiting investigation</h3><p>The analysis will appear here. If the API key is missing, the backend returns a configuration error.</p><div className="result-empty-meta"><span>Endpoint</span><code>POST /ai/analyze</code></div></div>}</Tile></div>
  </>;
}
function ModelPage({ model, health, loading, metrics, onRefresh }: { model: ModelInfo | null; health: Health | null; loading: boolean; metrics: Metrics | null; onRefresh: () => void }) {
  return <><PageHeading eyebrow="OPERATIONS / MODEL REGISTRY" title="Model details" description="Inspect the metadata and evaluation evidence for the active local artifact." action={<Button kind="secondary" renderIcon={RefreshCw} onClick={onRefresh}>Refresh metadata</Button>} />
    {loading ? <Tile className="panel loading-panel"><div className="loading-orbit" /><span>Loading active model metadata…</span></Tile> : model ? <><Tile className="panel model-hero"><div className="model-hero-symbol"><Layers3 size={24} /></div><div className="model-hero-copy"><div className="eyebrow">ACTIVE ARTIFACT <span className="status-inline-dot" /></div><h2>{model.algorithm || 'Machine learning model'}</h2><code>{model.model_version}</code><div className="model-tags"><Tag type="green">{health?.model_loaded ? 'LOADED' : 'NOT LOADED'}</Tag><Tag type="blue">{model.task || 'Classification'}</Tag><Tag type="cool-gray">{model.dataset?.type || 'Unknown dataset'}</Tag></div></div><div className="model-hero-date"><span>Created</span><strong>{dateLabel(model.created_at)}</strong></div></Tile>
      <div className="content-grid model-details-grid"><Tile className="panel"><SectionHeading title="Evaluation metrics" caption="Recorded at training time" /><div className="evaluation-list">{Object.entries(model.metrics || {}).map(([key, value]) => <div className="evaluation-row" key={key}><span>{metricLabel(key)}</span><div className="evaluation-bar"><span style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} /></div><strong>{value.toFixed(4)}</strong></div>)}</div><div className="panel-bottom-note"><CircleHelp size={14} /><span>These are held-out test metrics, not live performance estimates.</span></div></Tile><Tile className="panel"><SectionHeading title="Training record" caption="Artifact metadata" /><div className="metadata-list"><MetaRow label="Task" value={model.task || '—'} /><MetaRow label="Algorithm" value={model.algorithm || '—'} /><MetaRow label="Total training rows" value={model.training_rows?.toLocaleString() || '—'} /><MetaRow label="Train split" value={model.train_rows?.toLocaleString() || '—'} /><MetaRow label="Test split" value={model.test_rows?.toLocaleString() || '—'} /><MetaRow label="Feature count" value={String(model.features?.length ?? '—')} /><MetaRow label="Random state" value="42" /><MetaRow label="Dataset generator" value={model.dataset?.generator || '—'} /></div></Tile><Tile className="panel validation-panel"><SectionHeading title="Data validation" caption="Training-time contract check" action={<Tag type={model.data_validation?.valid ? 'green' : 'red'}>{model.data_validation?.valid ? 'VALID' : 'INVALID'}</Tag>} /><div className="validation-summary"><span className={`validation-mark ${model.data_validation?.valid ? 'validation-good' : 'validation-bad'}`}>{model.data_validation?.valid ? <CheckCircle2 size={20} /> : <TriangleAlert size={20} />}</span><div><strong>{model.data_validation?.valid ? 'Training data passed validation' : 'Training data did not pass validation'}</strong><p>{model.data_validation?.rows ?? model.training_rows ?? '—'} rows recorded in validation metadata.</p></div></div><div className="validation-foot"><span>Model server requests</span><strong>{metrics?.requests?.toLocaleString() ?? '—'}</strong><span>Prediction requests</span><strong>{metrics?.prediction_requests?.toLocaleString() ?? '—'}</strong></div></Tile></div></> : <Tile className="panel"><EmptyInline title="Model metadata unavailable" detail="Start the API and ensure a valid active model artifact exists." /></Tile>}
  </>;
}
function MetaRow({ label, value }: { label: string; value: string }) { return <div className="metadata-row"><span>{label}</span><strong>{value}</strong></div>; }
