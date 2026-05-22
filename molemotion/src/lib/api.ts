import type {
  ApiError,
  ApiResult,
  BatchPredictResponse,
  ChannelsResponse,
  EvaluateAgentResponse,
  FailureModesResponse,
  HealthResponse,
  KnowledgeListResponse,
  KnowledgeUploadResponse,
  PredictResponse,
  Variant,
  Verdict,
} from '../types.ts';

// In dev, Vite proxies /api → http://localhost:8000 (see vite.config.ts, task #21).
// In prod, the same-origin nginx in the frontend container proxies /api too.
// Override at build/runtime via VITE_API_BASE_URL when deploying separately.
const BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '');

const DEFAULT_TIMEOUT_MS = 30_000;
// /evaluate/agent runs Claude with up to 8 tool calls — real wall-clock
// is 30-120s depending on which tools the agent picks. 180s gives headroom;
// the spinner already covers the long-wait UX.
const AGENT_TIMEOUT_MS = 180_000;

interface RequestOpts extends Omit<RequestInit, 'signal'> {
  signal?: AbortSignal;
  timeoutMs?: number;
  parse?: 'json' | 'text';
}

function extractMessage(parsed: unknown, fallback: string): string {
  if (parsed && typeof parsed === 'object' && parsed !== null) {
    const obj = parsed as Record<string, unknown>;
    if (typeof obj.detail === 'string') return obj.detail;
    if (typeof obj.message === 'string') return obj.message;
    if (typeof obj.error === 'string') return obj.error;
  }
  return fallback;
}

async function request<T>(path: string, opts: RequestOpts = {}): Promise<ApiResult<T>> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: external, parse = 'json', headers, ...init } = opts;
  // FormData bodies must NOT have a manual Content-Type — the browser sets the
  // multipart boundary automatically.
  const isFormData = typeof FormData !== 'undefined' && init.body instanceof FormData;

  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new DOMException('timeout', 'AbortError')),
    timeoutMs,
  );
  const cleanup = () => clearTimeout(timer);
  const onExternalAbort = () => controller.abort(external?.reason);
  if (external) {
    if (external.aborted) controller.abort(external.reason);
    else external.addEventListener('abort', onExternalAbort, { once: true });
  }

  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: parse === 'text' ? 'text/plain' : 'application/json',
        ...(init.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
        ...(headers ?? {}),
      },
    });

    const text = await res.text();
    let body: unknown = text;
    if (parse === 'json' && text) {
      try { body = JSON.parse(text); } catch { body = text; }
    }

    if (!res.ok) {
      const error: ApiError = {
        status: res.status,
        message: extractMessage(body, res.statusText),
        detail: body,
      };
      return { ok: false, error };
    }

    return { ok: true, data: body as T };
  } catch (err) {
    const e = err as Error;
    const aborted = e?.name === 'AbortError';
    return {
      ok: false,
      error: {
        status: 0,
        message: aborted ? 'request aborted' : e?.message ?? 'network error',
      },
    };
  } finally {
    cleanup();
    if (external) external.removeEventListener('abort', onExternalAbort);
  }
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

export const api = {
  health(signal?: AbortSignal) {
    return request<HealthResponse>('/health', { signal });
  },

  pdbIds(opts: { q?: string; limit?: number; signal?: AbortSignal } = {}) {
    const params = new URLSearchParams();
    if (opts.q) params.set('q', opts.q);
    if (opts.limit != null) params.set('limit', String(opts.limit));
    const qs = params.toString();
    return request<string[]>(`/pdb_ids${qs ? `?${qs}` : ''}`, { signal: opts.signal });
  },

  predict(pdb_id: string, variant: Variant, signal?: AbortSignal) {
    return request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({ pdb_id, variant }),
      signal,
    });
  },

  predictBatch(pdb_ids: string[], variant: Variant, signal?: AbortSignal) {
    if (pdb_ids.length > 50) {
      return Promise.resolve<ApiResult<BatchPredictResponse>>({
        ok: false,
        error: { status: 413, message: 'max 50 ids per batch' },
      });
    }
    return request<BatchPredictResponse>('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ pdb_ids, variant }),
      signal,
      timeoutMs: 60_000,
    });
  },

  pdbString(
    pdb_id: string,
    opts: { stride?: number; dropWater?: boolean; signal?: AbortSignal } = {},
  ) {
    const stride = opts.stride ?? 5;
    const dropWater = opts.dropWater ?? true;
    return request<string>(
      `/pdb_string/${encodeURIComponent(pdb_id)}?stride=${stride}&drop_water=${dropWater}`,
      { signal: opts.signal, parse: 'text' },
    );
  },

  evaluate(pdb_id: string, variant: Variant, force = false, signal?: AbortSignal) {
    return request<Verdict>(`/evaluate?force=${force}`, {
      method: 'POST',
      body: JSON.stringify({ pdb_id, variant }),
      signal,
      timeoutMs: AGENT_TIMEOUT_MS,
    });
  },

  evaluateAgent(pdb_id: string, variant: Variant, force = false, signal?: AbortSignal) {
    return request<EvaluateAgentResponse>(`/evaluate/agent?force=${force}`, {
      method: 'POST',
      body: JSON.stringify({ pdb_id, variant }),
      signal,
      timeoutMs: AGENT_TIMEOUT_MS,
    });
  },

  failureModes(variant: Variant, signal?: AbortSignal) {
    return request<FailureModesResponse>(`/failure_modes?variant=${variant}`, { signal });
  },

  channels(pdb_id: string, signal?: AbortSignal) {
    return request<ChannelsResponse>(`/channels/${encodeURIComponent(pdb_id)}`, { signal });
  },

  uploadKnowledge(file: File, opts: { title?: string; pdbIds?: string; signal?: AbortSignal } = {}) {
    const form = new FormData();
    form.append('file', file);
    if (opts.title) form.append('title', opts.title);
    if (opts.pdbIds) form.append('pdb_ids', opts.pdbIds);
    return request<KnowledgeUploadResponse>('/knowledge/upload', {
      method: 'POST',
      body: form as any,
      signal: opts.signal,
      timeoutMs: 300_000,   // docling on a large PDF can take a minute+
    });
  },

  listKnowledge(signal?: AbortSignal) {
    return request<KnowledgeListResponse>('/knowledge', { signal });
  },

  deleteKnowledge(source_id: string, signal?: AbortSignal) {
    return request<{ source_id: string; chunks_removed: number }>(
      `/knowledge/${encodeURIComponent(source_id)}`,
      { method: 'DELETE', signal },
    );
  },
};

export type Api = typeof api;
