
import React, { useState, useEffect } from 'react';

export default function IACTDashboard() {
    const [concepts, setConcepts] = useState([]);
    const [title, setTitle] = useState('');
    const [category, setCategory] = useState('');
    const [keywords, setKeywords] = useState('');
    const [loading, setLoading] = useState(false);
    const [network, setNetwork] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResult, setSearchResult] = useState(null);
    const [searching, setSearching] = useState(false);
    const [telemetry, setTelemetry] = useState(null);
    const [sensory, setSensory] = useState({
        aModality: 'visual',
        aDescription: '',
        aFocus: '1.0',
        bModality: 'auditory',
        bDescription: '',
        bFocus: '1.0',
        delay: '1.0'
    });
    const [sensoryResult, setSensoryResult] = useState(null);
    const [sensoryLoading, setSensoryLoading] = useState(false);

    // Backend URL. On standalone local dev this defaults to the FastAPI port;
    // when the dashboard is served by the backend itself (e.g. Hugging Face
    // Space) main.jsx sets window.IACT_API_URL to "" for same-origin requests.
    const API_URL =
        typeof window !== "undefined" && window.IACT_API_URL !== undefined
            ? window.IACT_API_URL
            : "http://127.0.0.1:8000";

    // Fetch the network and live optimizer telemetry on load, then poll the
    // lightweight health endpoint so the CECANP-0.54 load indicator stays live.
    useEffect(() => {
        fetchNetwork();
        fetchHealth();
        const timer = setInterval(fetchHealth, 5000);
        return () => clearInterval(timer);
    }, []);

    const fetchNetwork = async () => {
        try {
            const response = await fetch(`${API_URL}/network/`);
            const data = await response.json();
            setNetwork(data.network || []);
        } catch (error) {
            console.error("Error fetching network:", error);
        }
    };

    const fetchHealth = async () => {
        try {
            const response = await fetch(`${API_URL}/health/`);
            const data = await response.json();
            setTelemetry(
                data.optimizer || {
                    compute_load: data.compute_load,
                    associative_bandwidth: data.associative_bandwidth,
                    cached_routes: data.cached_routes
                }
            );
        } catch (error) {
            console.error("Error fetching optimizer telemetry:", error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!title || !keywords) return;

        setLoading(true);
        const keywordArray = keywords.split(',').map(k => k.trim().toLowerCase());

        try {
            const response = await fetch(`${API_URL}/concepts/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title,
                    category: category || 'General',
                    keywords: keywordArray
                })
            });

            if (response.ok) {
                setTitle('');
                setCategory('');
                setKeywords('');
                fetchNetwork(); // Refresh network data
            }
        } catch (error) {
            console.error("Error saving concept:", error);
        } finally {
            setLoading(false);
        }
    };

    const updateSensory = (field, value) => {
        setSensory((current) => ({ ...current, [field]: value }));
    };

    // IACT-5SCP: submit two sensory inputs plus their capture delay.  Pairs
    // captured within 5 seconds are synthesised and anchored into the shared
    // ConceptStore, so they appear in the network and become searchable.
    const handleSensorySubmit = async (e) => {
        e.preventDefault();
        if (!sensory.aDescription.trim() || !sensory.bDescription.trim()) return;

        setSensoryLoading(true);
        try {
            const response = await fetch(`${API_URL}/optimizer/sensory-pair/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_a: {
                        modality: sensory.aModality,
                        description: sensory.aDescription,
                        focus_duration: Number(sensory.aFocus) || 0
                    },
                    item_b: {
                        modality: sensory.bModality,
                        description: sensory.bDescription,
                        focus_duration: Number(sensory.bFocus) || 0
                    },
                    capture_delay_seconds: Number(sensory.delay) || 0
                })
            });

            const data = await response.json();
            if (!response.ok) {
                setSensoryResult({
                    status: 'error',
                    message: data.detail || 'Sensory synthesis failed.'
                });
                return;
            }

            setSensoryResult(data);
            if (data.status === 'anchored') {
                setSensory((current) => ({
                    ...current,
                    aDescription: '',
                    bDescription: ''
                }));
                fetchNetwork();
                fetchHealth();
            }
        } catch (error) {
            console.error("Sensory synthesis failed:", error);
            setSensoryResult({
                status: 'error',
                message: 'Could not reach the sensory synthesis service.'
            });
        } finally {
            setSensoryLoading(false);
        }
    };

    // Cognitive search: asks the backend to resolve the query against the
    // associative network and returns {status, answer, concept_name,
    // confidence} (or a not_found / error status).
    const handleSearch = async (e) => {
        e.preventDefault();
        const query = searchQuery.trim();
        if (!query) return;

        setSearching(true);
        try {
            const response = await fetch(`${API_URL}/api/search/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            const data = await response.json();
            if (data.status === 'success') {
                setSearchResult({
                    status: 'success',
                    answer: data.answer,
                    concept: data.concept_name,
                    confidence: data.confidence
                });
            } else {
                setSearchResult({
                    status: data.status === 'error' ? 'error' : 'not_found',
                    message: data.message || 'No matching concept was found.'
                });
            }
        } catch (error) {
            console.error("Cognitive search failed:", error);
            setSearchResult({
                status: 'error',
                message: 'Could not reach the cognitive search service.'
            });
        } finally {
            setSearching(false);
        }
    };

    const computeLoad = Number(telemetry?.compute_load ?? 1.0);
    const associativeBandwidth = Number(telemetry?.associative_bandwidth ?? 0.0);
    const cachedRoutes = Number(telemetry?.cached_routes ?? 0);
    const loadTone =
        computeLoad <= 0.54
            ? 'bg-emerald-100 text-emerald-700'
            : computeLoad < 1.0
                ? 'bg-amber-100 text-amber-700'
                : 'bg-slate-100 text-slate-700';
    const anchoredConcepts = network.filter(
        (item) => item.category === 'Sensory Synthesis'
    );

    return (
        <div className="min-h-screen bg-slate-50 p-6 font-sans text-slate-900" dir="ltr">
            <div className="max-w-5xl mx-auto">

                {/* Header */}
                <header className="mb-8 border-b border-slate-200 pb-4">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <div>
                            <h1 className="text-2xl font-bold text-indigo-700">IACT Cognitive Dashboard</h1>
                            <p className="text-sm text-slate-600 mt-1">Integrative Associative Cognition Network Viewer</p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 text-xs">
                            <span className={`px-3 py-1.5 rounded-full font-semibold ${loadTone}`}>
                                Compute Load {computeLoad.toFixed(2)}
                            </span>
                            <span className="px-3 py-1.5 rounded-full font-semibold bg-indigo-100 text-indigo-700">
                                Associative Bandwidth {associativeBandwidth.toFixed(2)}
                            </span>
                            <span className="px-3 py-1.5 rounded-full font-medium bg-slate-100 text-slate-600">
                                {cachedRoutes} cached route{cachedRoutes === 1 ? '' : 's'}
                            </span>
                        </div>
                    </div>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                    {/* Cognitive Search Section */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 md:col-span-3">
                        <h2 className="text-lg font-semibold mb-4 text-slate-800">Cognitive Search</h2>
                        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Ask about a concept, e.g. what is artificial intelligence?"
                                className="flex-1 px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            />
                            <button
                                type="submit"
                                disabled={searching}
                                className="bg-slate-900 text-white px-5 py-2 rounded-xl font-medium hover:bg-slate-700 transition shadow-sm text-sm disabled:opacity-60"
                            >
                                {searching ? 'Searching…' : 'Search'}
                            </button>
                        </form>

                        {searchResult && (
                            <div className="mt-4 p-4 bg-slate-50 border border-slate-200 rounded-xl">
                                {searchResult.status === 'success' ? (
                                    <>
                                        <div className="flex items-center justify-between">
                                            <span className="font-bold text-indigo-700 text-sm">{searchResult.concept}</span>
                                            <span className="text-xs bg-indigo-100 text-indigo-700 px-2.5 py-1 rounded-full font-medium">
                                                {Math.round((searchResult.confidence ?? 0) * 100)}% match
                                            </span>
                                        </div>
                                        <p className="text-sm text-slate-700 mt-2">{searchResult.answer}</p>
                                    </>
                                ) : (
                                    <p className="text-sm text-slate-500 italic">
                                        {searchResult.message || 'No matching concept was found.'}
                                    </p>
                                )}
                            </div>
                        )}
                    </div>

                    {/* IACT-5SCP Sensory Pair Section */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 md:col-span-3">
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
                            <h2 className="text-lg font-semibold text-slate-800">5-Second Sensory Pair</h2>
                            <span className="text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full font-medium">
                                IACT-5SCP · volatile TTL 5.0s
                            </span>
                        </div>

                        <form onSubmit={handleSensorySubmit} className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Input A</p>
                                    <input
                                        type="text"
                                        value={sensory.aModality}
                                        onChange={(e) => updateSensory('aModality', e.target.value)}
                                        placeholder="Modality (e.g. visual)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                        required
                                    />
                                    <input
                                        type="text"
                                        value={sensory.aDescription}
                                        onChange={(e) => updateSensory('aDescription', e.target.value)}
                                        placeholder="Description (e.g. red square)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                        required
                                    />
                                    <input
                                        type="number"
                                        min="0"
                                        step="0.1"
                                        value={sensory.aFocus}
                                        onChange={(e) => updateSensory('aFocus', e.target.value)}
                                        placeholder="Focus duration (s)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                    />
                                </div>

                                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Input B</p>
                                    <input
                                        type="text"
                                        value={sensory.bModality}
                                        onChange={(e) => updateSensory('bModality', e.target.value)}
                                        placeholder="Modality (e.g. auditory)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                        required
                                    />
                                    <input
                                        type="text"
                                        value={sensory.bDescription}
                                        onChange={(e) => updateSensory('bDescription', e.target.value)}
                                        placeholder="Description (e.g. low tone)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                        required
                                    />
                                    <input
                                        type="number"
                                        min="0"
                                        step="0.1"
                                        value={sensory.bFocus}
                                        onChange={(e) => updateSensory('bFocus', e.target.value)}
                                        placeholder="Focus duration (s)"
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                    />
                                </div>
                            </div>

                            <div className="flex flex-col sm:flex-row sm:items-end gap-3">
                                <div className="sm:w-56">
                                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
                                        Capture Delay (s)
                                    </label>
                                    <input
                                        type="number"
                                        min="0"
                                        step="0.1"
                                        value={sensory.delay}
                                        onChange={(e) => updateSensory('delay', e.target.value)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                        required
                                    />
                                    <p className="text-xs text-slate-400 mt-1">≤ 5.0s anchors; &gt; 5.0s discards.</p>
                                </div>
                                <button
                                    type="submit"
                                    disabled={sensoryLoading}
                                    className="bg-indigo-600 text-white px-5 py-2 rounded-xl font-medium hover:bg-indigo-700 transition shadow-sm text-sm disabled:opacity-60"
                                >
                                    {sensoryLoading ? 'Synthesising…' : 'Synthesise Pair'}
                                </button>
                            </div>
                        </form>

                        {sensoryResult && (
                            <div className={`mt-4 p-4 rounded-xl border text-sm ${
                                sensoryResult.status === 'anchored'
                                    ? 'bg-emerald-50 border-emerald-200'
                                    : sensoryResult.status === 'discarded'
                                        ? 'bg-amber-50 border-amber-200'
                                        : 'bg-rose-50 border-rose-200'
                            }`}>
                                {sensoryResult.status === 'anchored' ? (
                                    <>
                                        <p className="font-semibold text-emerald-700">
                                            Anchored: {sensoryResult.concept?.concept}
                                        </p>
                                        <p className="text-slate-700 mt-1">{sensoryResult.concept?.description}</p>
                                        <p className="text-xs text-emerald-600 mt-2">
                                            Captured in {sensoryResult.capture_delay_seconds}s · volatile lock {sensoryResult.volatile_ttl_seconds}s
                                        </p>
                                    </>
                                ) : sensoryResult.status === 'discarded' ? (
                                    <p className="text-amber-700">{sensoryResult.reason}</p>
                                ) : (
                                    <p className="text-rose-600">
                                        {sensoryResult.message || sensoryResult.detail || 'Sensory synthesis failed.'}
                                    </p>
                                )}
                            </div>
                        )}

                        {anchoredConcepts.length > 0 && (
                            <div className="mt-5">
                                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                                    Anchored Concepts
                                </h3>
                                <div className="space-y-2">
                                    {anchoredConcepts.slice(-5).reverse().map((item, index) => (
                                        <div key={`${item.concept}-${index}`} className="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                                            <p className="font-semibold text-indigo-700 text-sm">{item.concept}</p>
                                            <p className="text-xs text-slate-600 mt-1">{item.description}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Form Section */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 md:col-span-1 h-fit">
                        <h2 className="text-lg font-semibold mb-4 text-slate-800">Add New Concept</h2>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Concept Title</label>
                                <input 
                                    type="text" 
                                    value={title} 
                                    onChange={(e) => setTitle(e.target.value)} 
                                    placeholder="e.g., Artificial Intelligence" 
                                    className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Category</label>
                                <input 
                                    type="text" 
                                    value={category} 
                                    onChange={(e) => setCategory(e.target.value)} 
                                    placeholder="e.g., Technology" 
                                    className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Keywords (Comma separated)</label>
                                <input 
                                    type="text" 
                                    value={keywords} 
                                    onChange={(e) => setKeywords(e.target.value)} 
                                    placeholder="ai, neural, cognition" 
                                    className="w-full px-3 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                                    required
                                />
                            </div>
                            <button 
                                type="submit" 
                                disabled={loading}
                                className="w-full bg-indigo-600 text-white py-2 rounded-xl font-medium hover:bg-indigo-700 transition shadow-sm text-sm"
                            >
                                {loading ? 'Integrating...' : 'Integrate into Network'}
                            </button>
                        </form>
                    </div>

                    {/* Network Visualization Section */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 md:col-span-2">
                        <h2 className="text-lg font-semibold mb-4 text-slate-800">Associative Network Map</h2>
                        
                        {network.length === 0 ? (
                            <p className="text-sm text-slate-400 italic">No concepts integrated yet. Add your first concept to build the network.</p>
                        ) : (
                            <div className="space-y-4">
                                {network.map((item, index) => (
                                    <div key={index} className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
                                        <div className="flex items-center justify-between">
                                            <h3 className="font-bold text-indigo-600 text-base">{item.concept}</h3>
                                            <span className="text-xs bg-indigo-100 text-indigo-700 px-2.5 py-1 rounded-full font-medium">
                                                {item.links ? item.links.length : 0} Connections
                                            </span>
                                        </div>
                                        
                                        <div className="mt-3 flex flex-wrap gap-2">
                                            {item.links && item.links.length > 0 ? (
                                                item.links.map((link, lIndex) => (
                                                    <span key={lIndex} className="text-xs bg-white border border-indigo-200 text-slate-700 px-3 py-1 rounded-lg shadow-2xs font-medium">
                                                        🔗 {link}
                                                    </span>
                                                ))
                                            ) : (
                                                <span className="text-xs text-slate-400 italic">Isolated node (No direct links yet)</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </div>
    );
}

