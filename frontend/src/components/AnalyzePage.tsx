
import { useState } from 'react'
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Loader2,
  RotateCcw,
  Sparkles,
  Upload,
} from 'lucide-react'

import { analyzeDocument } from '../services/api'
import type { PredictionResponse } from '../types/prediction'

function AnalyzePage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const wordCount = text.trim()
    ? text.trim().split(/\s+/).length
    : 0

  const canAnalyze = text.trim().length >= 20 &&
    text.length <= 50000 &&
    !loading

  async function handleAnalyze() {
    if (text.trim().length < 20) {
      setError('Please enter at least 20 characters.')
      return
    }

    if (text.length > 50000) {
      setError('Text cannot exceed 50,000 characters.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await analyzeDocument(text)
      setResult(response)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while analysing the document.'
      )
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setText('')
    setResult(null)
    setError('')
  }

  return (
    <div className="space-y-8">

      {/* Page heading */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
          <span className="text-[10px] font-semibold tracking-[0.18em] text-violet-300">
            INTELLIGENT TEXT ANALYSIS
          </span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-white">
          Analyse your document
        </h1>

        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
          Submit any text document and discover its underlying themes
          using three different topic modelling approaches.
        </p>
      </section>

      {/* Input section */}
      <section className="rounded-xl border border-white/[0.08] bg-[#0D1221] p-5 sm:p-7">

        <div className="mb-5 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-violet-500/10 p-2.5 text-violet-300">
              <FileText size={19} />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">
                Document input
              </h2>
              <p className="mt-1 text-[11px] text-slate-500">
                Paste or type your document below
              </p>
            </div>
          </div>

          <span className="hidden rounded-md border border-white/[0.08] px-2.5 py-1.5 text-[10px] text-slate-500 sm:block">
            TEXT DOCUMENT
          </span>
        </div>

        <textarea
          value={text}
          onChange={(event) => {
            setText(event.target.value)
            setError('')
            setResult(null)
          }}
          placeholder="Start typing or paste your document here...

For example: NASA has announced a new space mission to study planets and explore the solar system. Scientists are researching satellite technology and space exploration to understand the universe."
          maxLength={50000}
          className="min-h-[260px] w-full resize-y rounded-lg border border-white/[0.08] bg-[#080B16] p-5 text-sm leading-7 text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/10"
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex gap-4 text-[11px] text-slate-500">
            <span>{wordCount.toLocaleString()} words</span>
            <span>{text.length.toLocaleString()} / 50,000 characters</span>
          </div>

          <button
            onClick={() => {
              setText(
                'NASA has announced a new space mission to study planets and explore the solar system. Scientists are researching satellite technology and space exploration to understand the universe.'
              )
              setError('')
              setResult(null)
            }}
            className="flex items-center gap-1.5 text-[11px] text-violet-300 transition hover:text-violet-200"
          >
            <Upload size={13} />
            Load sample text
          </button>
        </div>

        {error && (
          <div className="mt-5 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-xs text-red-300">
            {error}
          </div>
        )}

        <div className="mt-6 flex flex-col-reverse gap-3 border-t border-white/[0.07] pt-5 sm:flex-row sm:items-center sm:justify-between">
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 rounded-lg border border-white/[0.08] px-4 py-3 text-xs text-slate-400 transition hover:bg-white/[0.04] hover:text-white"
          >
            <RotateCcw size={14} />
            Clear document
          </button>

          <button
            onClick={handleAnalyze}
            disabled={!canAnalyze}
            className="flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-6 py-3 text-xs font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Analysing document...
              </>
            ) : (
              <>
                <Sparkles size={16} />
                Analyse with AI
                <ArrowRight size={15} />
              </>
            )}
          </button>
        </div>
      </section>

      {/* Model information */}
      <section>
        <div className="mb-4">
          <h2 className="text-sm font-semibold text-white">
            Analysis pipeline
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Your document is evaluated independently by all three models.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {[
            {
              name: 'LDA',
              description: 'Probabilistic topic distribution',
              color: 'bg-blue-500',
            },
            {
              name: 'NMF',
              description: 'TF-IDF based topic extraction',
              color: 'bg-violet-500',
            },
            {
              name: 'BERTopic',
              description: 'Transformer-based semantic topics',
              color: 'bg-cyan-500',
            },
          ].map((model) => (
            <div
              key={model.name}
              className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-[#0D1221] p-4"
            >
              <div className={`h-2.5 w-2.5 rounded-full ${model.color}`} />
              <div>
                <p className="text-xs font-semibold text-white">
                  {model.name}
                </p>
                <p className="mt-1 text-[10px] text-slate-500">
                  {model.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Results */}
      {result && (
        <section className="space-y-5">

          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <div>
              <div className="flex items-center gap-2">
                <CheckCircle2 size={17} className="text-emerald-400" />
                <h2 className="text-lg font-semibold text-white">
                  Analysis complete
                </h2>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Prediction saved successfully · Record #{result.history_id}
              </p>
            </div>

            <span className="w-fit rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-[10px] text-emerald-400">
              SAVED TO DATABASE
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">

            {/* LDA result */}
            <div className="rounded-xl border border-white/[0.08] bg-[#0D1221] p-5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-white">LDA</span>
                <span className="rounded-md bg-blue-500/10 px-2 py-1 text-[10px] text-blue-300">
                  Probabilistic
                </span>
              </div>

              <p className="mt-6 text-[10px] tracking-wider text-slate-500">
                PREDICTED TOPIC
              </p>
              <p className="mt-2 text-3xl font-semibold text-white">
                #{result.predictions.lda.topic_id}
              </p>

              <div className="mt-6 flex justify-between text-xs">
                <span className="text-slate-400">Confidence</span>
                <span className="font-semibold text-blue-300">
                  {(result.predictions.lda.confidence * 100).toFixed(2)}%
                </span>
              </div>

              <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className="h-full rounded-full bg-blue-500"
                  style={{
                    width: `${Math.min(
                      result.predictions.lda.confidence * 100,
                      100
                    )}%`,
                  }}
                />
              </div>

              <p className="mt-5 text-[10px] text-slate-500">
                Topic distribution across 20 topics
              </p>
            </div>

            {/* NMF result */}
            <div className="rounded-xl border border-white/[0.08] bg-[#0D1221] p-5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-white">NMF</span>
                <span className="rounded-md bg-violet-500/10 px-2 py-1 text-[10px] text-violet-300">
                  Matrix factorization
                </span>
              </div>

              <p className="mt-6 text-[10px] tracking-wider text-slate-500">
                PREDICTED TOPIC
              </p>
              <p className="mt-2 text-3xl font-semibold text-white">
                #{result.predictions.nmf.topic_id}
              </p>

              <div className="mt-6 flex justify-between text-xs">
                <span className="text-slate-400">Confidence</span>
                <span className="font-semibold text-violet-300">
                  {(result.predictions.nmf.confidence * 100).toFixed(2)}%
                </span>
              </div>

              <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className="h-full rounded-full bg-violet-500"
                  style={{
                    width: `${Math.min(
                      result.predictions.nmf.confidence * 100,
                      100
                    )}%`,
                  }}
                />
              </div>

              <p className="mt-5 text-[10px] text-slate-500">
                Topic distribution across 20 topics
              </p>
            </div>

            {/* BERTopic result */}
            <div className="rounded-xl border border-white/[0.08] bg-[#0D1221] p-5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-white">
                  BERTopic
                </span>
                <span className="rounded-md bg-cyan-500/10 px-2 py-1 text-[10px] text-cyan-300">
                  Semantic
                </span>
              </div>

              <p className="mt-6 text-[10px] tracking-wider text-slate-500">
                PREDICTED TOPIC
              </p>

              <p className="mt-2 text-2xl font-semibold text-white">
                {result.predictions.bertopic.topic_name}
              </p>

              {result.predictions.bertopic.confidence !== null && (
                <>
                  <div className="mt-6 flex justify-between text-xs">
                    <span className="text-slate-400">Confidence</span>
                    <span className="font-semibold text-cyan-300">
                      {(
                        result.predictions.bertopic.confidence * 100
                      ).toFixed(2)}%
                    </span>
                  </div>

                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                    <div
                      className="h-full rounded-full bg-cyan-500"
                      style={{
                        width: `${Math.min(
                          result.predictions.bertopic.confidence * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </>
              )}

              {result.predictions.bertopic.topic_id === -1 && (
                <p className="mt-4 text-xs text-amber-400">
                  This document was classified as an outlier and was not
                  assigned to a discovered topic.
                </p>
              )}

              <div className="mt-5">
                <p className="mb-2 text-[10px] text-slate-500">
                  TOP KEYWORDS
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {result.predictions.bertopic.top_words.map((word) => (
                    <span
                      key={word}
                      className="rounded-md border border-cyan-500/15 bg-cyan-500/[0.06] px-2 py-1 text-[10px] text-cyan-200"
                    >
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Result summary */}
          <div className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5">
            <div className="flex items-center gap-2">
              <BrainCircuit size={17} className="text-violet-300" />
              <h3 className="text-sm font-semibold text-white">
                Prediction summary
              </h3>
            </div>

            <p className="mt-3 text-xs leading-6 text-slate-400">
              LDA identified topic {result.predictions.lda.topic_id},
              NMF identified topic {result.predictions.nmf.topic_id},
              and BERTopic identified{' '}
              {result.predictions.bertopic.topic_name}.
              {' '}Compare their outputs to understand how different
              modelling approaches interpret the same document.
            </p>
          </div>

          <button
            onClick={handleReset}
            className="flex items-center gap-2 rounded-lg border border-white/[0.08] px-4 py-3 text-xs text-slate-300 transition hover:bg-white/[0.04]"
          >
            <RotateCcw size={14} />
            Analyse another document
          </button>

        </section>
      )}
    </div>
  )
}

export default AnalyzePage
