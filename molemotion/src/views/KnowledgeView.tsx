import { useEffect, useState } from 'react';
import { Upload, FileText, Trash2, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

import { api } from '../lib/api.ts';
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
    <div className="flex flex-col gap-8 animate-in fade-in duration-300 max-w-5xl">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900 mb-2">Knowledge upload</h1>
        <p className="text-sm text-slate-500 leading-relaxed max-w-3xl">
          Add documents to the RAG corpus the independent agent uses. Supported via
          <span className="font-mono text-slate-700"> docling</span>:
          PDF, Word, PowerPoint, Excel, HTML, Markdown, plain text, and OCR'd images.
          Uploaded chunks are auto-tagged for label leaks — anything mentioning a Kd/Ki/IC50
          for a specific PDB is excluded when the agent queries that PDB.
        </p>
      </div>

      {/* ---------- Upload form ---------- */}
      <div className="border border-slate-200 rounded-lg bg-white shadow-sm overflow-hidden">
        <div className="bg-slate-50 border-b border-slate-200 px-6 py-4">
          <h2 className="font-semibold text-slate-800">Add a source</h2>
        </div>
        <div className="p-6 flex flex-col gap-4">
          <div
            onDragOver={e => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('know-file')?.click()}
            className={`border-2 border-dashed rounded-md p-8 text-center cursor-pointer transition-colors ${
              dragActive
                ? 'border-indigo-400 bg-indigo-50'
                : file
                  ? 'border-emerald-300 bg-emerald-50/30'
                  : 'border-slate-300 bg-slate-50 hover:bg-slate-100'
            }`}
          >
            <input
              id="know-file"
              type="file"
              accept={ACCEPT}
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
            {file ? (
              <div className="flex items-center justify-center gap-3">
                <FileText size={28} className="text-emerald-600" />
                <div className="text-left">
                  <div className="font-medium text-slate-800">{file.name}</div>
                  <div className="text-xs text-slate-500">
                    {(file.size / 1024).toFixed(1)} KB · {file.type || 'unknown type'}
                  </div>
                </div>
                <button
                  onClick={e => { e.stopPropagation(); setFile(null); }}
                  className="ml-4 text-xs text-slate-500 hover:text-rose-600 underline"
                >
                  remove
                </button>
              </div>
            ) : (
              <div className="text-slate-500">
                <Upload size={28} className="mx-auto mb-2 text-slate-400" />
                <div className="font-medium">Drop a file or click to choose</div>
                <div className="text-xs text-slate-400 mt-1">
                  {ACCEPT.replace(/\./g, '').replace(/,/g, ', ')}
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-slate-700">Title</span>
              <input
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder={file?.name ?? 'paper title or source name'}
                className="border border-slate-200 rounded px-3 py-2 font-mono text-xs focus:outline-none focus:border-indigo-400"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-slate-700">
                PDB IDs <span className="text-slate-400 font-normal">(optional, comma-separated)</span>
              </span>
              <input
                type="text"
                value={pdbIds}
                onChange={e => setPdbIds(e.target.value)}
                placeholder="1A1B, 3BL7"
                className="border border-slate-200 rounded px-3 py-2 font-mono text-xs focus:outline-none focus:border-indigo-400"
              />
            </label>
          </div>

          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-500 italic">
              {file && (file.size > 5_000_000
                ? 'large file — docling may take up to a minute'
                : 'docling will extract structured Markdown'
              )}
            </p>
            <button
              disabled={!file || uploading}
              onClick={handleUpload}
              className="bg-indigo-600 text-white text-sm font-semibold px-5 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {uploading && <Loader2 size={14} className="animate-spin" />}
              {uploading ? 'Uploading + embedding…' : 'Upload'}
            </button>
          </div>

          {uploadError && (
            <div className="border border-rose-200 bg-rose-50 text-rose-800 text-sm px-4 py-3 rounded flex items-start gap-2">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div>
                <div className="font-medium">Upload failed</div>
                <div className="text-xs font-mono mt-1">{uploadError.message} (status {uploadError.status})</div>
              </div>
            </div>
          )}

          {uploadResult && (
            <div className="border border-emerald-200 bg-emerald-50 text-emerald-800 text-sm px-4 py-3 rounded flex items-start gap-2">
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
                  <div className="text-xs text-emerald-700 mt-1">
                    No leak signals detected — chunks will return for any PDB query.
                  </div>
                )}
                {uploadResult.warning && (
                  <div className="text-xs text-amber-700 mt-1">⚠ {uploadResult.warning}</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ---------- Existing sources ---------- */}
      <div className="border border-slate-200 rounded-lg bg-white shadow-sm overflow-hidden">
        <div className="bg-slate-50 border-b border-slate-200 px-6 py-4 flex items-center justify-between">
          <h2 className="font-semibold text-slate-800">Uploaded sources</h2>
          <button
            onClick={refreshSources}
            className="text-xs text-slate-500 hover:text-slate-700 underline"
          >
            refresh
          </button>
        </div>
        {listLoading && (
          <div className="p-6 text-sm text-slate-500 flex items-center gap-2">
            <Loader2 size={14} className="animate-spin" /> loading…
          </div>
        )}
        {!listLoading && sources.length === 0 && (
          <div className="p-6 text-sm text-slate-400 italic">
            No uploaded sources yet — drop a paper above.
          </div>
        )}
        {sources.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/50 text-slate-500 text-xs uppercase">
                <th className="text-left font-medium py-2 px-4">Title</th>
                <th className="text-left font-medium py-2 px-4">Kind</th>
                <th className="text-right font-medium py-2 px-4">Chunks</th>
                <th className="text-left font-medium py-2 px-4">Uploaded</th>
                <th className="text-right font-medium py-2 px-4"></th>
              </tr>
            </thead>
            <tbody>
              {sources.map(s => (
                <tr key={s.source_id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/40">
                  <td className="py-2.5 px-4 font-medium text-slate-700 truncate max-w-[300px]" title={s.title}>{s.title}</td>
                  <td className="py-2.5 px-4 font-mono text-xs text-slate-500">{s.kind}</td>
                  <td className="py-2.5 px-4 text-right font-mono text-slate-600">{s.n_chunks}</td>
                  <td className="py-2.5 px-4 text-xs font-mono text-slate-500">{s.uploaded_at}</td>
                  <td className="py-2.5 px-4 text-right">
                    <button
                      onClick={() => handleDelete(s.source_id)}
                      disabled={deleting === s.source_id}
                      className="text-slate-400 hover:text-rose-600 p-1 disabled:opacity-50"
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
        )}
      </div>
    </div>
  );
}
