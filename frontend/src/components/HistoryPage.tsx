import { useCallback, useEffect, useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  BrainCircuit,
  CalendarDays,
  ChevronRight,
  Clock3,
  FileText,
  Loader2,
  Search,
  X,
} from 'lucide-react'

import {
  getPredictionHistory,
  getPredictionById,
} from '../services/api'

import type {
  PredictionHistoryRecord,
} from '../types/prediction'

function HistoryPage() {
  const [records, setRecords] = useState<PredictionHistoryRecord[]>([])
  const [selectedRecord, setSelectedRecord] =
    useState<PredictionHistoryRecord | null>(null)

  const [search, setSearch] = useState('')
  const [limit] = useState(10)
  const [offset, setOffset] = useState(0)
  const [total, setTotal] = useState(0)

  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState('')

  const loadHistory = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const response = await getPredictionHistory(limit, offset)
      setRecords(response.history)
      setTotal(response.total_records)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to retrieve prediction history.'
      )
    } finally {
      setLoading(false)
    }
  }, [limit, offset])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  async function openRecord(id: number) {
    setDetailLoading(true)
    setError('')

    try {
      const response = await getPredictionById(id)
      setSelectedRecord(response.history)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to retrieve this record.'
      )
    } finally {
      setDetailLoading(false)
    }
  }

  const filteredRecords = records.filter((record) => {
    const query = search.toLowerCase()

    return (
      record.input_text.toLowerCase().includes(query) ||
      String(record.id).includes(query)
    )
  })

  const totalPages = Math.max(1, Math.ceil(total / limit))
  const currentPage = Math.floor(offset / limit) + 1

  function formatDate(date: string) {
    return new Date(date).toLocaleString()
  }

  return (
    <div className="space-y-8">

      {/* Heading */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
          <span className="text-[10px] font-semibold tracking-[0.18em] text-violet-300">
            ANALYSIS ARCHIVE
          </span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-white">
          Prediction History
        </h1>

        <p className="mt-3 text-sm text-slate-400">
          Access your previously analysed documents and compare model results.
        </p>
      </section>

      {/* Statistics */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5">
          <div className="flex items-center gap-2 text-slate-400">
            <FileText size={17} />
            <span className="text-xs">Total predictions saved</span>
          </div>

          <p className="mt-4 text-3xl font-semibold text-white">
            {total.toLocaleString()}
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5">
          <div className="flex items-center gap-2 text-slate-400">
            <BrainCircuit size={17} />
            <span className="text-xs">Models per prediction</span>
          </div>

          <p className="mt-4 text-3xl font-semibold text-white">
            03
          </p>
          <p className="mt-1 text-[10px] text-slate-500">
            LDA · NMF · BERTopic
          </p>
        </div>
      </section>

      {/* Search */}
      <section className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5">

        <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-sm font-semibold text-white">
              Previous analyses
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Select a record to view its complete results.
            </p>
          </div>

          <div className="relative sm:w-72">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search ID or document text..."
              className="w-full rounded-lg border border-white/[0.08] bg-[#080B16] py-2.5 pl-9 pr-3 text-xs text-white outline-none placeholder:text-slate-600 focus:border-violet-500/50"
            />
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center gap-3 py-16 text-sm text-slate-400">
            <Loader2 size={18} className="animate-spin" />
            Loading prediction history...
          </div>
        ) : error ? (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
            {error}
            <button
              onClick={loadHistory}
              className="ml-3 underline"
            >
              Retry
            </button>
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="py-16 text-center">
            <FileText size={30} className="mx-auto text-slate-600" />
            <p className="mt-4 text-sm text-slate-300">
              No prediction records found.
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Try another search or analyse a new document.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredRecords.map((record) => (
              <button
                key={record.id}
                onClick={() => openRecord(record.id)}
                className="group w-full rounded-lg border border-white/[0.07] bg-white/[0.015] p-4 text-left transition hover:border-violet-500/30 hover:bg-violet-500/[0.04]"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-md bg-violet-500/10 px-2 py-1 text-[10px] font-semibold text-violet-300">
                        RECORD #{record.id}
                      </span>

                      <span className="flex items-center gap-1 text-[10px] text-slate-500">
                        <CalendarDays size={12} />
                        {formatDate(record.created_at)}
                      </span>
                    </div>

                    <p className="mt-3 line-clamp-2 text-xs leading-6 text-slate-300">
                      {record.input_text}
                    </p>

                    <div className="mt-4 flex flex-wrap gap-2">
                      <span className="rounded-md border border-blue-500/20 bg-blue-500/[0.06] px-2 py-1 text-[10px] text-blue-300">
                        LDA · #{record.lda_prediction.topic_id}
                      </span>

                      <span className="rounded-md border border-violet-500/20 bg-violet-500/[0.06] px-2 py-1 text-[10px] text-violet-300">
                        NMF · #{record.nmf_prediction.topic_id}
                      </span>

                      <span className="rounded-md border border-cyan-500/20 bg-cyan-500/[0.06] px-2 py-1 text-[10px] text-cyan-300">
                        BERTopic · {record.bertopic_prediction.topic_name}
                      </span>
                    </div>
                  </div>

                  <ChevronRight
                    size={18}
                    className="mt-1 shrink-0 text-slate-600 transition group-hover:translate-x-1 group-hover:text-violet-300"
                  />
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Pagination */}
        {!loading && total > 0 && (
          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.07] pt-5">
            <p className="text-[11px] text-slate-500">
              Showing {offset + 1}–
              {Math.min(offset + limit, total)} of {total} records
            </p>

            <div className="flex items-center gap-3">
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
                className="rounded-lg border border-white/[0.08] p-2 text-slate-300 transition hover:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ArrowLeft size={15} />
              </button>

              <span className="text-xs text-slate-400">
                Page {currentPage} of {totalPages}
              </span>

              <button
                disabled={offset + limit >= total}
                onClick={() => setOffset(offset + limit)}
                className="rounded-lg border border-white/[0.08] p-2 text-slate-300 transition hover:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Record detail modal */}
      {(selectedRecord || detailLoading) && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center overflow-y-auto bg-black/75 p-4 backdrop-blur-sm">
          <div className="my-auto max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-white/10 bg-[#0D1221] p-5 shadow-2xl sm:p-7">

            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] tracking-[0.18em] text-violet-300">
                  PREDICTION DETAILS
                </p>
                <h2 className="mt-2 text-lg font-semibold text-white">
                  {selectedRecord
                    ? `Record #${selectedRecord.id}`
                    : 'Loading record...'}
                </h2>
              </div>

              <button
                onClick={() => setSelectedRecord(null)}
                className="rounded-lg border border-white/10 p-2 text-slate-400 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>

            {detailLoading ? (
              <div className="flex justify-center py-16 text-slate-400">
                <Loader2 size={22} className="animate-spin" />
              </div>
            ) : selectedRecord && (
              <div className="mt-6 space-y-6">
                <div>
                  <div className="mb-2 flex items-center gap-2 text-xs text-slate-400">
                    <Clock3 size={14} />
                    {formatDate(selectedRecord.created_at)}
                  </div>

                  <div className="rounded-lg border border-white/[0.07] bg-[#080B16] p-4">
                    <p className="mb-2 text-[10px] tracking-wider text-slate-500">
                      ORIGINAL DOCUMENT
                    </p>
                    <p className="whitespace-pre-wrap text-xs leading-6 text-slate-300">
                      {selectedRecord.input_text}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div className="rounded-lg border border-blue-500/20 bg-blue-500/[0.05] p-4">
                    <p className="text-xs font-semibold text-blue-300">LDA</p>
                    <p className="mt-3 text-2xl font-semibold text-white">
                      #{selectedRecord.lda_prediction.topic_id}
                    </p>
                    <p className="mt-2 text-xs text-slate-400">
                      {(selectedRecord.lda_prediction.confidence * 100).toFixed(2)}% confidence
                    </p>
                  </div>

                  <div className="rounded-lg border border-violet-500/20 bg-violet-500/[0.05] p-4">
                    <p className="text-xs font-semibold text-violet-300">NMF</p>
                    <p className="mt-3 text-2xl font-semibold text-white">
                      #{selectedRecord.nmf_prediction.topic_id}
                    </p>
                    <p className="mt-2 text-xs text-slate-400">
                      {(selectedRecord.nmf_prediction.confidence * 100).toFixed(2)}% confidence
                    </p>
                  </div>

                  <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.05] p-4">
                    <p className="text-xs font-semibold text-cyan-300">BERTopic</p>
                    <p className="mt-3 text-lg font-semibold text-white">
                      {selectedRecord.bertopic_prediction.topic_name}
                    </p>
                    <p className="mt-2 text-xs text-slate-400">
                      {selectedRecord.bertopic_prediction.confidence === null
                        ? 'Confidence unavailable'
                        : `${(selectedRecord.bertopic_prediction.confidence * 100).toFixed(2)}% confidence`}
                    </p>
                  </div>
                </div>

                <div>
                  <p className="mb-3 text-xs font-semibold text-white">
                    BERTopic keywords
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {selectedRecord.bertopic_prediction.top_words.map((word) => (
                      <span
                        key={word}
                        className="rounded-md border border-cyan-500/20 bg-cyan-500/[0.06] px-3 py-1.5 text-xs text-cyan-200"
                      >
                        {word}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="flex justify-end border-t border-white/[0.07] pt-5">
                  <button
                    onClick={() => setSelectedRecord(null)}
                    className="rounded-lg bg-violet-600 px-5 py-2.5 text-xs font-semibold text-white hover:bg-violet-500"
                  >
                    Close details
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default HistoryPage