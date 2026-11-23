import { useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  Bell,
  BrainCircuit,
  ChevronDown,
  Clock3,
  Command,
  Database,
  FileText,
  History,
  LayoutDashboard,
  Layers3,
  Menu,
  Plus,
  Search,
  Settings,
  Sparkles,
  X,
} from 'lucide-react'

import AnalyzePage from './components/AnalyzePage'
import HistoryPage from './components/HistoryPage'
import TopicsPage from './components/TopicsPage'

const navigation = [
  { label: 'Dashboard', icon: LayoutDashboard },
  { label: 'Analyze Text', icon: FileText },
  { label: 'Prediction History', icon: History },
  { label: 'Topics Explorer', icon: Layers3 },
]

function App() {
  const [activePage, setActivePage] = useState('Dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const stats = [
    {
      label: 'Documents Analyzed',
      value: '18,810',
      note: 'Available in training dataset',
      icon: FileText,
    },
    {
      label: 'Active Models',
      value: '03',
      note: 'LDA, NMF and BERTopic',
      icon: BrainCircuit,
    },
    {
      label: 'Discovered Topics',
      value: '87',
      note: 'BERTopic discovered topics',
      icon: Layers3,
    },
    {
      label: 'BERTopic Coverage',
      value: '70.06%',
      note: 'Training evaluation coverage',
      icon: Activity,
    },
  ]

  const models = [
    {
      name: 'Latent Dirichlet Allocation',
      shortName: 'LDA',
      description: 'Probabilistic topic modelling',
      coherence: '0.5828',
      color: 'bg-blue-500',
    },
    {
      name: 'Non-negative Matrix Factorization',
      shortName: 'NMF',
      description: 'Matrix decomposition based modelling',
      coherence: '0.7194',
      color: 'bg-violet-500',
    },
    {
      name: 'BERTopic',
      shortName: 'BERTopic',
      description: 'Transformer-based topic modelling',
      coherence: '0.6459',
      color: 'bg-cyan-500',
    },
  ]

  return (
    <div className="min-h-screen bg-[#080B16] text-slate-100">

      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-white/[0.07] bg-[#0B0F1C] transition-transform duration-300 lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand */}
        <div className="flex h-20 items-center justify-between border-b border-white/[0.07] px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-600/15 text-violet-400">
              <BrainCircuit size={23} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                TopicLens<span className="text-violet-400"> AI</span>
              </h1>
              <p className="text-[10px] tracking-[0.18em] text-slate-500">
                INTELLIGENCE PLATFORM
              </p>
            </div>
          </div>

          <button
            onClick={() => setSidebarOpen(false)}
            className="text-slate-500 lg:hidden"
          >
            <X size={20} />
          </button>
        </div>

        {/* Workspace */}
        <div className="px-4 pt-7">
          <p className="mb-3 px-3 text-[10px] font-semibold tracking-[0.2em] text-slate-500">
            WORKSPACE
          </p>

          <nav className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = activePage === item.label

              return (
                <button
                  key={item.label}
                  onClick={() => {
                    setActivePage(item.label)
                    setSidebarOpen(false)
                  }}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm transition ${
                    isActive
                      ? 'border border-violet-500/20 bg-violet-500/10 text-violet-300'
                      : 'text-slate-400 hover:bg-white/[0.04] hover:text-white'
                  }`}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                  {isActive && (
                    <span className="ml-auto h-1.5 w-1.5 rounded-full bg-violet-400" />
                  )}
                </button>
              )
            })}
          </nav>
        </div>

        {/* System */}
        <div className="mt-8 px-4">
          <p className="mb-3 px-3 text-[10px] font-semibold tracking-[0.2em] text-slate-500">
            PREFERENCES
          </p>

          <button
            onClick={() => setActivePage('Settings')}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm transition ${
              activePage === 'Settings'
                ? 'bg-violet-500/10 text-violet-300'
                : 'text-slate-400 hover:bg-white/[0.04] hover:text-white'
            }`}
          >
            <Settings size={18} />
            Settings
          </button>
        </div>

        {/* Bottom card */}
        <div className="mt-auto p-4">
          <div className="rounded-xl border border-violet-500/15 bg-gradient-to-br from-violet-500/10 to-transparent p-4">
            <div className="mb-3 flex items-center gap-2 text-violet-300">
              <Sparkles size={16} />
              <span className="text-xs font-semibold">AI Workspace</span>
            </div>
            <p className="text-xs leading-5 text-slate-400">
              Explore your documents and discover hidden themes with three
              topic modelling approaches.
            </p>
            <p className="mt-4 text-[10px] text-slate-600">
              TOPICLENS AI · VERSION 1.0
            </p>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="min-h-screen lg:ml-64">

        {/* Topbar */}
        <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-white/[0.07] bg-[#080B16]/90 px-5 backdrop-blur-xl sm:px-8">

          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-slate-400 lg:hidden"
            >
              <Menu size={22} />
            </button>

            <div>
              <p className="text-xs text-slate-500">Workspace / Overview</p>
              <h2 className="mt-1 text-sm font-semibold text-white">
                {activePage}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-5">
            <button className="hidden items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-xs text-slate-500 sm:flex">
              <Search size={14} />
              Search
              <span className="ml-4 flex items-center gap-1 rounded border border-white/10 px-1.5 py-0.5">
                <Command size={10} /> K
              </span>
            </button>

            <button className="relative text-slate-400 hover:text-white">
              <Bell size={19} />
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-violet-500" />
            </button>

            <div className="hidden h-7 w-px bg-white/10 sm:block" />

            <button className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-violet-400/20 bg-violet-500/10 text-xs font-semibold text-violet-300">
                TL
              </div>
              <ChevronDown size={14} className="hidden text-slate-500 sm:block" />
            </button>
          </div>
        </header>

        {/* Dashboard content */}
        <main className="mx-auto max-w-[1500px] p-5 sm:p-8">

          {activePage === 'Analyze Text' ? (
            <AnalyzePage />
          ) : activePage === 'Prediction History' ? (
            <HistoryPage />
          ) : activePage === 'Topics Explorer' ? (
            <TopicsPage />
          ) : (
            <>

              {/* Welcome section */}
              <section className="mb-9 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    <span className="text-[10px] font-semibold tracking-[0.18em] text-emerald-400">
                      WORKSPACE OVERVIEW
                    </span>
                  </div>

                  <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    Discover what your data is saying.
                  </h1>

                  <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
                    Analyse documents, uncover hidden topics and compare results
                    across multiple machine learning models.
                  </p>
                </div>

                <button
                  onClick={() => setActivePage('Analyze Text')}
                  className="flex shrink-0 items-center justify-center gap-2 rounded-lg bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/20 transition hover:bg-violet-500"
                >
                  <Plus size={17} />
                  New Analysis
                </button>
              </section>

              {/* Statistics */}
              <section className="mb-9 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {stats.map((stat) => {
                  const Icon = stat.icon

                  return (
                    <div
                      key={stat.label}
                      className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5 transition hover:border-violet-500/20"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-slate-400">
                          {stat.label}
                        </span>
                        <div className="rounded-lg bg-violet-500/10 p-2 text-violet-300">
                          <Icon size={17} />
                        </div>
                      </div>

                      <p className="mt-5 text-3xl font-semibold tracking-tight text-white">
                        {stat.value}
                      </p>

                      <p className="mt-2 text-[11px] text-slate-500">
                        {stat.note}
                      </p>
                    </div>
                  )
                })}
              </section>

              {/* Models section */}
              <section className="mb-9">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-semibold text-white">
                      Model Performance
                    </h3>
                    <p className="mt-1 text-xs text-slate-500">
                      Topic coherence scores from Phase 5 evaluation
                    </p>
                  </div>

                  <span className="rounded-md border border-white/[0.08] px-3 py-1.5 text-[10px] text-slate-400">
                    C_v COHERENCE
                  </span>
                </div>

                <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
                  {models.map((model) => (
                    <div
                      key={model.shortName}
                      className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-5"
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className={`h-2 w-2 rounded-full ${model.color}`} />
                            <span className="text-sm font-semibold text-white">
                              {model.shortName}
                            </span>
                          </div>
                          <p className="mt-2 text-[11px] text-slate-500">
                            {model.description}
                          </p>
                        </div>

                        <ArrowUpRight size={17} className="text-slate-600" />
                      </div>

                      <div className="mt-7 flex items-end justify-between">
                        <div>
                          <p className="text-[10px] tracking-wider text-slate-500">
                            COHERENCE SCORE
                          </p>
                          <p className="mt-1 text-3xl font-semibold text-white">
                            {model.coherence}
                          </p>
                        </div>
                        <div className="mb-1 text-right">
                          <p className="text-[10px] text-slate-500">Model</p>
                          <p className="text-xs text-slate-300">
                            {model.shortName}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Bottom section */}
              <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">

                {/* Quick analysis */}
                <div className="rounded-xl border border-violet-500/20 bg-gradient-to-br from-violet-500/[0.10] to-[#0D1221] p-6 xl:col-span-2">
                  <div className="flex items-center gap-2 text-violet-300">
                    <Sparkles size={17} />
                    <span className="text-xs font-semibold">
                      START EXPLORING
                    </span>
                  </div>

                  <h3 className="mt-5 text-xl font-semibold text-white">
                    Ready to discover hidden topics?
                  </h3>

                  <p className="mt-2 max-w-lg text-sm leading-6 text-slate-400">
                    Submit a text document and let LDA, NMF and BERTopic analyse
                    it independently. Compare their predictions in one place.
                  </p>

                  <button
                    onClick={() => setActivePage('Analyze Text')}
                    className="mt-6 flex items-center gap-2 rounded-lg border border-violet-400/30 bg-violet-500/15 px-4 py-2.5 text-xs font-semibold text-violet-200 transition hover:bg-violet-500/25"
                  >
                    Analyse a document
                    <ArrowUpRight size={15} />
                  </button>
                </div>

                {/* System information */}
                <div className="rounded-xl border border-white/[0.07] bg-[#0D1221] p-6">
                  <h3 className="text-sm font-semibold text-white">
                    System Overview
                  </h3>

                  <div className="mt-5 space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Database size={16} className="text-violet-300" />
                        <span className="text-xs text-slate-400">Database</span>
                      </div>
                      <span className="flex items-center gap-1.5 text-[10px] text-emerald-400">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                        Verified
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <BrainCircuit size={16} className="text-violet-300" />
                        <span className="text-xs text-slate-400">ML Models</span>
                      </div>
                      <span className="text-[10px] text-emerald-400">
                        Loaded in backend
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Clock3 size={16} className="text-violet-300" />
                        <span className="text-xs text-slate-400">API</span>
                      </div>
                      <span className="text-[10px] text-slate-400">
                        Integration pending
                      </span>
                    </div>
                  </div>

                  <div className="mt-6 border-t border-white/[0.07] pt-4">
                    <p className="text-[10px] text-slate-500">
                      Backend integration will be enabled in the next step.
                    </p>
                  </div>
                </div>
              </section>

              {/* Footer */}
              <footer className="mt-10 flex flex-col justify-between gap-2 border-t border-white/[0.07] pt-5 text-[10px] text-slate-600 sm:flex-row">
                <span>TopicLens AI · Intelligent Topic Modelling</span>
                <span>Built with React, FastAPI and Machine Learning</span>
              </footer>

            </>
          )}

        </main>
      </div>
    </div>
  )
}

export default App