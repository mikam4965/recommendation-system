/**
 * RecSys Dashboard Application
 */

import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  SystemMetrics,
  ModelComparison,
  ModelMetricsTable,
  FunnelVisualization,
  ABTestResults,
} from './components';
import {
  LayoutDashboard,
  BarChart3,
  FlaskConical,
  Funnel,
  Settings,
  Menu,
  X,
} from 'lucide-react';
import { clsx } from 'clsx';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    },
  },
});

type TabId = 'overview' | 'models' | 'experiments' | 'funnel';

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

const TABS: Tab[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="w-5 h-5" /> },
  { id: 'models', label: 'Models', icon: <BarChart3 className="w-5 h-5" /> },
  { id: 'experiments', label: 'Experiments', icon: <FlaskConical className="w-5 h-5" /> },
  { id: 'funnel', label: 'Funnel', icon: <Funnel className="w-5 h-5" /> },
];

function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg lg:hidden"
            >
              {sidebarOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>
            <h1 className="text-xl font-bold text-gray-900">
              RecSys Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              RetailRocket Dataset
            </span>
            <button className="p-2 hover:bg-gray-100 rounded-lg">
              <Settings className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={clsx(
            'fixed lg:static inset-y-0 left-0 z-20 w-64 bg-white border-r border-gray-200 transform transition-transform lg:transform-none',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          )}
        >
          <nav className="p-4 space-y-1 mt-16 lg:mt-0">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  if (window.innerWidth < 1024) {
                    setSidebarOpen(false);
                  }
                }}
                className={clsx(
                  'flex items-center gap-3 w-full px-4 py-3 rounded-lg text-left transition-colors',
                  activeTab === tab.id
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50'
                )}
              >
                {tab.icon}
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-6 lg:p-8 min-h-screen">
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'models' && <ModelsTab />}
          {activeTab === 'experiments' && <ExperimentsTab />}
          {activeTab === 'funnel' && <FunnelTab />}
        </main>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-10 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Overview</h2>
        <p className="text-gray-500 mt-1">
          System metrics and performance summary
        </p>
      </div>

      <SystemMetrics />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelComparison />
        <FunnelVisualization />
      </div>
    </div>
  );
}

function ModelsTab() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Model Performance</h2>
        <p className="text-gray-500 mt-1">
          Compare recommendation model metrics
        </p>
      </div>

      <ModelComparison />
      <ModelMetricsTable />
    </div>
  );
}

function ExperimentsTab() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">A/B Experiments</h2>
        <p className="text-gray-500 mt-1">
          Manage and analyze recommendation experiments
        </p>
      </div>

      <ABTestResults />
    </div>
  );
}

function FunnelTab() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Conversion Funnel</h2>
        <p className="text-gray-500 mt-1">
          User journey analysis and conversion metrics
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FunnelVisualization />
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Key Insights
          </h3>
          <ul className="space-y-3 text-sm text-gray-600">
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">•</span>
              <span>
                Cart-to-transaction rate (31.9%) indicates high purchase intent
                once items are added to cart
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-500 mt-1">•</span>
              <span>
                View-to-cart conversion (2.6%) suggests opportunity for
                improved product recommendations
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span>
                Long-tail distribution: 80% of interactions with top 1% of items
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-500 mt-1">•</span>
              <span>
                Funnel-aware models show +11% NDCG improvement over static approaches
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}

export default App;
