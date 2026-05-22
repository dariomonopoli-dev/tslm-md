// UI-side types used across views.

export type Variant = 'v1a' | 'v1b';

export type RecommendationLabel = 'trust' | 'review' | 'discard';

export type ClaimStatus = 'verified' | 'contradicted' | 'unverifiable';

export interface TriageRow {
  pdb: string;
  pred: number;
  delta: number;
  regex: string;
  recommendation: RecommendationLabel;
  evidence: string;
}

export interface FailurePattern {
  cluster: string;
  count: number;
  systems: string;
}

export interface FailureModeRow {
  pdb: string;
  model: number;
  vina: number;
  mmgbsa: number;
  agent: RecommendationLabel;
  reason: string;
}

export interface ScoreData {
  label: string;
  score: number;
  description: string;
}

// ---------------------------------------------------------------------------
// Backend response types — match FRONTEND_v2.md §8 and §9.4 exactly.
// ---------------------------------------------------------------------------

export interface VerifierClaim {
  text: string;
  status: ClaimStatus;
  evidence: string;
}

export interface RegexVerifierResult {
  verified: number;
  contradicted: number;
  unverifiable: number;
  total: number;
  claims: VerifierClaim[];
}

export interface PredictResponse {
  pdb_id: string;
  variant: Variant;
  pK: number;
  rationale: string;
  hidden_pK: number | null;
  regex_verifier: RegexVerifierResult;
  latency_ms: number;
  model_version: string;
  rag_corpus_version: string;
  judge_model: string;
  // Tunnel backend extras (optional — only present when INFERENCE_BACKEND=tunnel)
  verdict?: 'CONFIRMED' | 'INCONCLUSIVE' | string;
  verdict_reason?: string;
  confidence?: 'low' | 'medium' | 'high' | string;
  affinity?: number;            // kcal/mol (the tunnel's raw output)
  independent_energy?: number;
  disagreement_z?: number;
  channel_summary?: Record<string, {
    start: number;
    end: number;
    mean: number;
    std: number;
    min: number;
    max: number;
    trend: 'increasing' | 'decreasing' | 'stable' | 'fluctuating' | string;
  }>;
}

export interface BatchPredictFailure {
  pdb_id: string;
  error: string;
}

export interface BatchPredictResponse {
  results: PredictResponse[];
  failed: BatchPredictFailure[];
}

export interface HealthResponse {
  status: 'ready' | 'starting';
  variants_loaded: string[];
  warm_since: string | null;
  rag_corpus_version: string;
  judge_model: string;
  spend_today_usd?: number;
  remaining_cap_usd?: number;
}

export interface VerdictScores {
  structural_consistency: number;
  physical_consistency: number;
  literature_consistency: number;
  chemical_plausibility: number;
}

export interface AgentTraceUsage {
  tool_calls: number;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
}

export interface Verdict {
  scores: VerdictScores;
  verified_claims: { claim: string; evidence: string }[];
  contradicted_claims: { claim: string; contradicting_evidence: string }[];
  missing_claims: { evidence: string; why_relevant: string }[];
  recommendation: RecommendationLabel;
  citations: { chunk_id: string; score: number }[];
  independence_caveats: string[];
  judge_model: string;
  rag_corpus_version: string;
  tool_versions?: Record<string, string>;
  agent_trace: AgentTraceUsage;
  cached?: boolean;
}

export interface TraceStep {
  step: number;
  tool: string;
  input: Record<string, unknown>;
  result: Record<string, unknown>;
  latency_ms: number;
}

export interface EvaluateAgentResponse {
  verdict: Verdict;
  trace: TraceStep[];
}

export interface FailureModesResponse {
  variant: Variant;
  rows: FailureModeRow[];
  patterns: FailurePattern[];
  generated_at: string;
}

export interface ChannelFrame {
  frame: number;
  rmsd: number;
  energy: number;
  dist: number;
  bsasa: number;
}

export interface ChannelsResponse {
  pdb_id: string;
  n_frames: number;
  frames: ChannelFrame[];
}

export interface KnowledgeUploadResponse {
  source_id: string;
  title: string;
  kind: string;
  filename?: string;
  chunks_added: number;
  n_label_chunks?: number;
  pdb_ids_tagged?: string[];
  text_length_chars?: number;
  warning?: string;
}

export interface KnowledgeSource {
  source_id: string;
  title: string;
  kind: string;
  uploaded_at: string;
  n_chunks: number;
}

export interface KnowledgeListResponse {
  sources: KnowledgeSource[];
}

// ---------------------------------------------------------------------------
// Generic result envelope returned by the API client.
// Components branch on `ok` instead of try/catch.
// ---------------------------------------------------------------------------

export interface ApiError {
  status: number;       // HTTP status, or 0 for network/abort
  message: string;
  detail?: unknown;
}

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ApiError };
