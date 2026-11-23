import { useCallback, useEffect, useState } from 'react'
import {
  BrainCircuit,
  ChevronRight,
  FileText,
  Layers3,
  Loader2,
  Search,
  X,
} from 'lucide-react'

import { getTopics, getTopicById } from '../services/api'
import type { Topic } from '../types/prediction'

type ModelType = 'lda' | 'nmf' | 'bertopic'

const MODEL_INFO: Record<ModelType, { name: string; shortName: string; count: string; color: string; border: string; bg: string }> = {
  lda: {
    name: 'Latent Dirichlet Allocation',
    shortName: 'LDA',
    count: '20 topics',
    color: 'text-blue-400',
    border: 'border-blue-500/30',
    bg: 'bg-blue-500/10',
  },
  nmf: {
    name: 'Non-negative Matrix Factorization',
    shortName: 'NMF',
    count: '20 topics',
    color: 'text-violet-400',
    border: 'border-violet-500/30',
    bg: 'bg-violet-500/10',
  },
  bertopic: {
    name: 'Transformer-based Topic Modelling',
    shortName: 'BERTopic',
    count: '87 discovered topics',
    color: 'text-cyan-400',
    border: 'border-cyan-500/30',
    bg: 'bg-cyan-500/10',
  },
}

function TopicsPage() {
  const [model, setModel] = useState<ModelType>('lda')
  const [search, setSearch] = useState('')
  const [includeOutliers, setIncludeOutliers] = useState(false)

  const [topics, setTopics] = useState<Topic[]>([])
  const [totalTopics, setTotalTopics] = useState(0)

  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null)
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchTopics = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const response = await getTopics(model, search, includeOutliers)
      setTopics(response.topics)
      setTotalTopics(response.total_topics)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to retrieve topics.'
      )
      setTopics([])
      setTotalTopics(0)
    } finally {
      setLoading(false)
    }
  }, [model, search, includeOutliers])

  useEffect(() => {
    fetchTopics()
  }, [fetchTopics])

  const handleTopicClick = async (topicId: number) => {
    setDetailLoading(true)
    setError('')

    try {
      const response = await getTopicById(model, topicId)
      setSelectedTopic(response.topic)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to load topic details.'
      )
    } finally {
      setDetailLoading(false)
    }
  }

  const activeInfo = MODEL_INFO[model]

  return (
    <div className="space-y-8">
      {/* Header */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
          <span className="text-[10px] font-semibold tracking-[0.18em] text-violet-300">
            NLP KNOWLEDGE BASE
          </span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-white">
          Topics Explorer
        </h1>

        <p className="mt-3 text-sm text-slate-400">
          Interactively browse, filter, and inspect document clusters across all three topic modelling algorithms.
        </p>
      </section>

      {/* Model Cards */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {(['lda', 'nmf', 'bertopic'] as ModelType[]).map((mKey) => {
          const info = MODEL_INFO[mKey]
          const isSelected = model === mKey

          return (
            <button
              key={mKey}
              onClick={() => {
                setModel(mKey)
                setSelectedTopic(null)
              }}
              className={`rounded-xl border p-5 text-left transition ${
                isSelected
                  ? `${info.border} ${info.bg} ring-1 ring-violet-500/30`
                  : 'border-white/[0.07] bg-[#0D1221] hover:border-white/20'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold uppercase tracking-wider ${info.color}`}>
                  {info.shortName}
                </span>
                {isSelected && (
                  <span className="h-2 w-2 rounded-full bg-violet-400" />
                )}
              </div>

              <h3 className="mt-2 text-sm font-semibold text-white">
                {info.name}
              </h3>

              <p className="mt-4 text-xs font-medium text-slate-400">
                {info.count}
              </p>
            </button>
          )
        })}
      </section>

      {/* Search and Filters */}
      <section className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative flex-1">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by topic name or keywords..."
              className="w-full rounded-lg border border-white/[0.08] bg-[#080B16] py-2.5 pl-9 pr-3 text-xs text-white outline-none placeholder:text-slate-600 focus:border-violet-500/50"
            />
          </div>

          {model === 'bertopic' && (
            <label className="flex items-center gap-2 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={includeOutliers}
                onChange={(e) => setIncludeOutliers(e.target.checked)}
                className="h-4 w-4 rounded border-white/10 bg-[#080B16] accent-violet-600"
              />
              Include BERTopic Outliers (-1)
            </label>
          )}

          <div className="flex items-center gap-2 text-xs text-slate-400 shrink-0">
            <Layers3 size={15} className="text-violet-400" />
            <span>Showing <strong className="text-white">{totalTopics}</strong> topics</span>
          </div>
        </div>

        {/* Content State */}
        <div className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center gap-3 py-16 text-sm text-slate-400">
              <Loader2 size={18} className="animate-spin text-violet-400" />
              Fetching {activeInfo.shortName} topics...
            </div>
          ) : error ? (
            <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
              {error}
              <button onClick={fetchTopics} className="ml-3 underline">
                Retry
              </button>
            </div>
          ) : topics.length === 0 ? (
            <div className="py-16 text-center">
              <FileText size={30} className="mx-auto text-slate-600" />
              <p className="mt-4 text-sm text-slate-300">No matching topics found.</p>
              <p className="mt-1 text-xs text-slate-500">
                Try refining your keyword search.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {topics.map((topic) => (
                <button
                  key={topic.topic_id}
                  onClick={() => handleTopicClick(topic.topic_id)}
                  className="group rounded-xl border border-white/[0.07] bg-white/[0.015] p-5 text-left transition hover:border-violet-500/30 hover:bg-violet-500/[0.04]"
                >
                  <div className="flex items-center justify-between">
                    <span className="rounded-md bg-violet-500/10 px-2.5 py-1 text-[10px] font-semibold text-violet-300">
                      {topic.is_outlier ? 'OUTLIER' : `TOPIC #${topic.topic_id}`}
                    </span>

                    {topic.document_count !== null && (
                      <span className="text-[11px] text-slate-500">
                        {topic.document_count.toLocaleString()} docs
                      </span>
                    )}
                  </div>

                  <h3 className="mt-3 line-clamp-1 text-sm font-semibold text-white group-hover:text-violet-200">
                    {topic.topic_name}
                  </h3>

                  <div className="mt-4 flex flex-wrap gap-1.5">
                    {topic.keywords.slice(0, 6).map((kw, idx) => (
                      <span
                        key={`${kw}-${idx}`}
                        className="rounded border border-white/[0.06] bg-[#080B16] px-2 py-0.5 text-[10px] text-slate-400"
                      >
                        {kw}
                      </span>
                    ))}
                  </div>

                  <div className="mt-5 flex items-center justify-between text-xs font-medium text-violet-400">
                    <span>Explore details</span>
                    <ChevronRight size={15} className="transition group-hover:translate-x-1" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Details Modal */}
      {(selectedTopic || detailLoading) && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-white/10 bg-[#0D1221] p-6 shadow-2xl sm:p-7">
            <div className="flex items-center justify-between border-b border-white/[0.07] pb-4">
              <div>
                <p className="text-[10px] font-semibold tracking-wider text-violet-300 uppercase">
                  {model} TOPIC DETAILS
                </p>
                <h2 className="mt-1 text-lg font-semibold text-white">
                  {selectedTopic ? selectedTopic.topic_name : 'Loading topic...'}
                </h2>
              </div>

              <button
                onClick={() => setSelectedTopic(null)}
                className="rounded-lg border border-white/10 p-2 text-slate-400 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>

            {detailLoading ? (
              <div className="flex justify-center py-16 text-slate-400">
                <Loader2 size={22} className="animate-spin" />
              </div>
            ) : selectedTopic && (
              <div className="mt-6 space-y-6">
                {selectedTopic.document_count !== null && (
                  <div className="flex items-center gap-3 rounded-lg border border-white/[0.07] bg-[#080B16] p-4">
                    <BrainCircuit className="text-cyan-400" size={20} />
                    <div>
                      <p className="text-[10px] text-slate-500 uppercase">Document Volume</p>
                      <p className="text-base font-semibold text-white">
                        {selectedTopic.document_count.toLocaleString()} documents assigned
                      </p>
                    </div>
                  </div>
                )}

                <div>
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Keywords
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedTopic.keywords.map((kw, idx) => (
                      <span
                        key={`${kw}-${idx}`}
                        className="rounded-md border border-violet-500/20 bg-violet-500/10 px-3 py-1.5 text-xs text-violet-300"
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>

                {selectedTopic.representative_docs && selectedTopic.representative_docs.length > 0 && (
                  <div>
                    <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Representative Sample Documents
                    </h3>
                    <div className="space-y-3">
                      {selectedTopic.representative_docs.map((doc, idx) => (
                        <div
                          key={idx}
                          className="rounded-lg border border-white/[0.07] bg-[#080B16] p-4 text-xs leading-6 text-slate-300"
                        >
                          <span className="mb-2 block text-[10px] font-semibold text-violet-400">
                            SAMPLE #{idx + 1}
                          </span>
                          {doc}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex justify-end border-t border-white/[0.07] pt-4">
                  <button
                    onClick={() => setSelectedTopic(null)}
                    className="rounded-lg bg-violet-600 px-5 py-2 text-xs font-semibold text-white hover:bg-violet-500"
                  >
                    Close
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

export default TopicsPage