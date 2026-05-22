import { useEffect, useState } from 'react';
import { Upload, FileText, Trash2, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

import { api } from '../lib/api.ts';
import { DarkInput } from '../components/ui.tsx';
import { GlowCard, CardHeader } from '../components/GlowCard.tsx';
import type {
  ApiError,
  KnowledgeSource,
  KnowledgeUploadResponse,
} from '../types.ts';


const ACCEPT = '.pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.html,.htm,.md,.markdown,.txt,.png,.jpg,.jpeg';


export function KnowledgeView() {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [pdbIds, setPdbIds] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<KnowledgeUploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<ApiError | null>(null);

  const [sources, setSources] = useState<KnowledgeSource[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  async function refreshSources() {
    setListLoading(true);
    const res = await api.listKnowledge();
    setListLoading(false);
    if (res.ok) setSources(res.data.sources);
  }

  useEffect(() => {
    refreshSources();
  }, []);

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    const res = await api.uploadKnowledge(file, {
      title: title || file.name,
      pdbIds,
    });
    setUploading(false);
    if (!res.ok) {
      setUploadError(res.error);
      return;
    }
    setUploadResult(res.data);
    setFile(null);
    setTitle('');
    setPdbIds('');
    refreshSources();
  }

  async function handleDelete(source_id: string) {
    if (!confirm('Remove this source and all its chunks from the RAG corpus?')) return;
    setDeleting(source_id);
    await api.deleteKnowledge(source_id);
    setDeleting(null);
    refreshSources();
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  }

  return (
    <div className="flex flex-col gap-6 max-w-5xl">
      <div className="fx-fade-up">
        <span className="text-[11px] font-mono tracking-[0.22em] uppercase"
              style={{ color: 'var(--color-ink-dim)' }}>
          Knowledge
        </span>
        <h1 className="font-display text-3xl font-semibold tracking-[-0.025em] mt-1 mb-3"
            style={{ color: 'var(--color-ink)' }}>
          Curate the agent’s evidence corpus
        </h1>
        <p className="text-[15px] leading-relaxed max-w-3xl"
           style={{ color: 'var(--color-ink-mute)' }}>
          Add documents to the RAG corpus the independent agent uses. Supported via
          {' '}<span className="font-mono" style={{ color: 'var(--color-ink-2)' }}>docling</span>:
          PDF, Word, PowerPoint, Excel, HTML, Markdown, plain text, and OCR’d images.
          Uploaded chunks are auto-tagged for label leaks — anything mentioning a Kd/Ki/IC50
          for a specific PDB is excluded when the agent queries that PDB.
        </p>
      </div>

      {/* Upload card */}
      <GlowCard className="fx-fade-up" style={{ animationDelay: '60ms' } as React.CSSProperties}>
        <CardHeader>Add a source</CardHeader>
        <div className="p-6 flex flex-col gap-4">
          <div
            onDragOver={e => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('know-file')?.click()}
            className="rounded-2xl p-10 text-center cursor-pointer transition-all relative overflow-hidden bevel-border"
            style={{
              background: dragActive
                ? 'rgba(84, 103, 242, 0.12)'
                : file
                  ? 'rgba(78, 192, 122, 0.08)'
                  : '#fafbfd',
              border: dragActive
                ? '2px dashed #5467F2'
                : file
                  ? '2px dashed rgba(78, 192, 122, 0.6)'
                  : '2px dashed var(--color-line-strong)',
            }}
          >
            <input
              id="know-file"
              type="file"
              accept={ACCEPT}
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
            {file ? (
              <div className="flex items-center justify-center gap-4">
                <FileText size={32} style={{ color: '#4ec07a' }} />
                <div className="text-left">
                  <div className="font-medium" style={{ color: 'var(--color-ink)' }}>{file.name}</div>
                  <div className="text-xs font-mono" style={{ color: 'var(--color-ink-dim)' }}>
                    {(file.size / 1024).toFixed(1)} KB · {file.type || 'unknown type'}
                  </div>
                </div>
                <button
                  onClick={e => { e.stopPropagation(); setFile(null); }}
                  className="ml-4 text-xs underline transition-colors hover:text-[#f4625f]"
                  style={{ color: 'var(--color-ink-dim)' }}
                >
                  remove
                </button>
              </div>
            ) : (
              <div style={{ color: 'var(--color-ink-mute)' }}>
                <Upload size={32} className="mx-auto mb-3" style={{ color: '#5467F2' }} />
                <div className="font-display font-medium text-base"
                     style={{ color: 'var(--color-ink)' }}>
                  Drop a file or click to choose
                </div>
                <div className="text-[11px] font-mono mt-2 tracking-wider"
                     style={{ color: 'var(--color-ink-dim)' }}>
                  {ACCEPT.replace(/\./g, '').replace(/,/g, ' · ')}
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-[10px] font-mono tracking-[0.2em] uppercase"
                    style={{ color: 'var(--color-ink-dim)' }}>
                Title
              </span>
              <DarkInput
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder={file?.name ?? 'paper title or source name'}
              />
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-[10px] font-mono tracking-[0.2em] uppercase"
                    style={{ color: 'var(--color-ink-dim)' }}>
                PDB IDs · optional, comma-separated
              </span>
              <DarkInput
                type="text"
                value={pdbIds}
                onChange={e => setPdbIds(e.target.value)}
                placeholder="1A1B, 3BL7"
              />
            </label>
          </div>

          <div className="flex items-center justify-between gap-3 flex-wrap">
            <p className="text-xs italic" style={{ color: 'var(--color-ink-dim)' }}>
              {file && (file.size > 5_000_000
                ? 'large file — docling may take up to a minute'
                : 'docling will extract structured Markdown'
              )}
            </p>
            <button
              disabled={!file || uploading}
              onClick={handleUpload}
              className="btn-primary rounded-lg px-5 py-2 text-sm flex items-center gap-2"
            >
              {uploading && <Loader2 size={14} className="animate-spin" />}
              {uploading ? 'Uploading + embedding…' : 'Upload'}
            </button>
          </div>

          {uploadError && (
            <div className="rounded-xl px-4 py-3 text-sm flex items-start gap-2"
                 style={{
                   background: 'rgba(244, 98, 95, 0.12)',
                   border: '1px solid rgba(244, 98, 95, 0.4)',
                   color: '#a8332f',
                 }}>
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div>
                <div className="font-medium">Upload failed</div>
                <div className="text-xs font-mono mt-1">{uploadError.message} (status {uploadError.status})</div>
              </div>
            </div>
          )}

          {uploadResult && (
            <div className="rounded-xl px-4 py-3 text-sm flex items-start gap-2"
                 style={{
                   background: 'rgba(78, 192, 122, 0.1)',
                   border: '1px solid rgba(78, 192, 122, 0.4)',
                   color: '#2d8b56',
                 }}>
              <CheckCircle2 size={16} className="shrink-0 mt-0.5" />
              <div>
                <div className="font-medium">
                  Added {uploadResult.chunks_added} chunks ({uploadResult.text_length_chars ?? 0} chars from {uploadResult.kind})
                </div>
                {uploadResult.n_label_chunks ? (
                  <div className="text-xs font-mono mt-1">
                    {uploadResult.n_label_chunks} chunks flagged contains_label
                    {uploadResult.pdb_ids_tagged?.length
                      ? ` — leak-filtered for: ${uploadResult.pdb_ids_tagged.join(', ')}`
                      : ''}
                  </div>
                ) : (
                  <div className="text-xs mt-1" style={{ color: '#2d8b56' }}>
                    No leak signals detected — chunks will return for any PDB query.
                  </div>
                )}
                {uploadResult.warning && (
                  <div className="text-xs mt-1" style={{ color: '#FF9900' }}>
                    ⚠ {uploadResult.warning}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </GlowCard>

      {/* Sources list */}
      <GlowCard className="fx-fade-up" style={{ animationDelay: '120ms' } as React.CSSProperties}>
        <CardHeader
          right={
            <button
              onClick={refreshSources}
              className="text-[11px] font-mono underline transition-colors hover:text-ink"
              style={{ color: 'var(--color-ink-dim)' }}
            >
              refresh
            </button>
          }
        >
          Uploaded sources
        </CardHeader>
        {listLoading && (
          <div className="p-6 text-sm flex items-center gap-2"
               style={{ color: 'var(--color-ink-mute)' }}>
            <Loader2 size={14} className="animate-spin" /> loading…
          </div>
        )}
        {!listLoading && sources.length === 0 && (
          <div className="p-6 text-sm italic"
               style={{ color: 'var(--color-ink-dim)' }}>
            No uploaded sources yet — drop a paper above.
          </div>
        )}
        {sources.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] font-mono tracking-[0.18em] uppercase"
                    style={{ color: 'var(--color-ink-dim)' }}>
                  <th className="text-left font-medium py-2.5 px-4 border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Title</th>
                  <th className="text-left font-medium py-2.5 px-4 border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Kind</th>
                  <th className="text-right font-medium py-2.5 px-4 border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Chunks</th>
                  <th className="text-left font-medium py-2.5 px-4 border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Uploaded</th>
                  <th className="text-right font-medium py-2.5 px-4 border-b"
                      style={{ borderColor: 'var(--color-line)' }}></th>
                </tr>
              </thead>
              <tbody>
                {sources.map((s, i) => (
                  <tr key={s.source_id}
                      className="fx-rise hover:bg-[rgba(84,103,242,0.06)] transition-colors"
                      style={{
                        animationDelay: `${i * 30}ms`,
                        borderBottom: '1px solid var(--color-line)',
                      }}>
                    <td className="py-2.5 px-4 font-medium truncate max-w-[300px]"
                        style={{ color: 'var(--color-ink)' }}
                        title={s.title}>{s.title}</td>
                    <td className="py-2.5 px-4 font-mono text-xs"
                        style={{ color: 'var(--color-ink-mute)' }}>{s.kind}</td>
                    <td className="py-2.5 px-4 text-right font-mono"
                        style={{ color: 'var(--color-ink-2)' }}>{s.n_chunks}</td>
                    <td className="py-2.5 px-4 text-xs font-mono"
                        style={{ color: 'var(--color-ink-mute)' }}>{s.uploaded_at}</td>
                    <td className="py-2.5 px-4 text-right">
                      <button
                        onClick={() => handleDelete(s.source_id)}
                        disabled={deleting === s.source_id}
                        className="p-1 transition-colors hover:text-[#f4625f]"
                        style={{ color: 'var(--color-ink-dim)' }}
                        title="remove from RAG corpus"
                      >
                        {deleting === s.source_id
                          ? <Loader2 size={14} className="animate-spin" />
                          : <Trash2 size={14} />}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlowCard>
    </div>
  );
}
