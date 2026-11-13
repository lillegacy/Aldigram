
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

    // Backend URL. On standalone local dev this defaults to the FastAPI port;
    // when the dashboard is served by the backend itself (e.g. Hugging Face
    // Space) main.jsx sets window.IACT_API_URL to "" for same-origin requests.
    const API_URL =
        typeof window !== "undefined" && window.IACT_API_URL !== undefined
            ? window.IACT_API_URL
            : "http://127.0.0.1:8000";

    // Fetch existing network on load
    useEffect(() => {
        fetchNetwork();
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

    return (
        <div className="min-h-screen bg-slate-50 p-6 font-sans text-slate-900" dir="ltr">
            <div className="max-w-5xl mx-auto">
                
                {/* Header */}
                <header className="mb-8 border-b border-slate-200 pb-4">
                    <h1 className="text-2xl font-bold text-indigo-700">IACT Cognitive Dashboard</h1>
                    <p className="text-sm text-slate-600 mt-1">Integrative Associative Cognition Network Viewer</p>
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

