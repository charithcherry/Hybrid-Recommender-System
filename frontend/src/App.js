/**
 * Main App Component - Fashion Recommendation System
 * Features: Sidebar navigation, Filters, Tab-based views, Split recommendations
 */

import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

import UserLogin from './components/UserLogin';
import Sidebar from './components/Sidebar';
import FilterBar from './components/FilterBar';
import AllProducts from './components/AllProducts';
import ProductGrid from './components/ProductGrid';
import Chat from './components/Chat';
import {
  getColdStartRecommendations,
  getSplitRecommendations,
  trackInteraction as apiTrackInteraction,
  getUserProfile,
  getUserInteractionStates,
} from './services/api';

function App() {
  // User state
  const [userId, setUserId] = useState(null);
  const [username, setUsername] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userProfile, setUserProfile] = useState(null);
  const [interactionStates, setInteractionStates] = useState({});

  // UI state
  const [activeTab, setActiveTab] = useState('all'); // 'all', 'popular', 'foryou'
  const [filters, setFilters] = useState({
    gender: null,
    masterCategory: null,
    subCategory: null,
    articleType: null,
    baseColour: null,
    season: null,
    usage: null
  });

  // Product state
  const [popularProducts, setPopularProducts] = useState([]);
  const [cfRecommendations, setCfRecommendations] = useState([]);
  const [contentRecommendations, setContentRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  // Check for existing session on mount
  useEffect(() => {
    const storedUserId = localStorage.getItem('userId');
    const storedUsername = localStorage.getItem('username');

    if (storedUserId && storedUsername) {
      setUserId(parseInt(storedUserId));
      setUsername(storedUsername);
      setIsLoggedIn(true);
    }
  }, []);

  // Load user profile and interaction states when logged in
  useEffect(() => {
    if (userId) {
      loadUserProfile();
      loadInteractionStates();
    }
  }, [userId]);

  // Load data when tab or filters change
  useEffect(() => {
    if (isLoggedIn) {
      if (activeTab === 'popular') {
        loadPopularProducts();
      } else if (activeTab === 'foryou') {
        loadRecommendations();
      }
    }
  }, [activeTab, filters, isLoggedIn]);

  const loadUserProfile = async () => {
    try {
      const profile = await getUserProfile(userId);
      setUserProfile(profile);
    } catch (error) {
      console.error('Error loading profile:', error);
    }
  };

  const loadInteractionStates = async () => {
    try {
      const states = await getUserInteractionStates(userId);
      setInteractionStates(states.interaction_states || {});
    } catch (error) {
      console.error('Error loading interaction states:', error);
    }
  };

  const loadPopularProducts = async () => {
    setLoading(true);
    try {
      const response = await getColdStartRecommendations(50, filters);
      setPopularProducts(response.recommendations);
    } catch (error) {
      console.error('Error loading popular products:', error);
      toast.error('Failed to load popular products');
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async () => {
    if (!userId) return;

    setLoading(true);
    try {
      const response = await getSplitRecommendations(userId, 10, filters);
      setCfRecommendations(response.cf_recommendations || []);
      setContentRecommendations(response.content_recommendations || []);
    } catch (error) {
      console.error('Error loading recommendations:', error);
      toast.error('Failed to load recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleInteraction = async (itemId, type) => {
    if (!userId) {
      toast.error('Please log in first');
      return;
    }

    try {
      const response = await apiTrackInteraction(userId, itemId, type);

      // Update interaction states based on response
      const newStates = { ...interactionStates };
      if (response.action === 'removed') {
        delete newStates[itemId];
        toast.info(`${type} removed`);
      } else if (response.action === 'added') {
        newStates[itemId] = type;
        const messages = {
          like: 'Liked ❤️',
          dislike: 'Disliked 💔',
          save: 'Saved 🔖',
          buy: 'Added to cart 🛒'
        };
        toast.success(messages[type] || 'Interaction recorded');
      } else if (response.action === 'updated') {
        newStates[itemId] = type;
        toast.info(`Changed to ${type}`);
      }

      setInteractionStates(newStates);

      // Reload user profile to update interaction count
      await loadUserProfile();

      // Reload recommendations if on that tab
      if (activeTab === 'foryou') {
        await loadRecommendations();
      }
    } catch (error) {
      console.error('Error tracking interaction:', error);
      toast.error('Failed to record interaction');
    }
  };

  const handleLoginSuccess = (userData) => {
    setUserId(userData.user_id);
    setUsername(userData.username);
    setIsLoggedIn(true);

    // Store in localStorage
    localStorage.setItem('userId', userData.user_id.toString());
    localStorage.setItem('username', userData.username);

    toast.success(`Welcome, ${userData.username}!`);
    setActiveTab('popular'); // Start with popular tab
  };

  const handleLogout = () => {
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    setUserId(null);
    setUsername('');
    setIsLoggedIn(false);
    setPopularProducts([]);
    setCfRecommendations([]);
    setContentRecommendations([]);
    setUserProfile(null);
    setInteractionStates({});
    toast.info('Logged out successfully');
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  if (!isLoggedIn) {
    return <UserLogin onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-container">
      <ToastContainer position="top-right" autoClose={3000} />

      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={{ username }}
        interactionCount={userProfile?.total_interactions || 0}
        onLogout={handleLogout}
      />

      {/* Main Content Area */}
      <div className="main-content">
        {/* Header with App Title */}
        <div className="app-header">
          <h1 className="app-title">Fashion Recommender 🛍️</h1>
        </div>

        {/* Filter Bar (hide on chat tab) */}
        {activeTab !== 'chat' && (
          <FilterBar filters={filters} onFilterChange={handleFilterChange} />
        )}

        {/* Content Area */}
        <div className="content-area">
          {activeTab === 'all' && (
            <AllProducts
              filters={filters}
              onInteraction={handleInteraction}
              interactionStates={interactionStates}
            />
          )}

          {activeTab === 'popular' && (
            <div className="popular-tab">
              <div className="tab-header">
                <h2>🔥 Top 50 Popular Products</h2>
                <p className="tab-description">
                  Most loved items by our community
                </p>
              </div>
              {loading ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Loading popular products...</p>
                </div>
              ) : (
                <ProductGrid
                  products={popularProducts}
                  onInteraction={handleInteraction}
                  interactionStates={interactionStates}
                />
              )}
            </div>
          )}

          {activeTab === 'foryou' && (
            <div className="foryou-tab">
              <div className="tab-header">
                <h2>✨ Personalized For You</h2>
                <p className="tab-description">
                  Recommendations based on your preferences
                </p>
              </div>

              {loading ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Generating recommendations...</p>
                </div>
              ) : (
                <>
                  {/* Content-Based Section (FIRST - Primary recommendations) */}
                  {contentRecommendations.length > 0 && (
                    <div className="recommendation-section">
                      <div className="section-header">
                        <h3>🎯 Similar Items</h3>
                        <p>Based on items you've liked</p>
                      </div>
                      <ProductGrid
                        products={contentRecommendations}
                        onInteraction={handleInteraction}
                        interactionStates={interactionStates}
                      />
                    </div>
                  )}

                  {/* Collaborative Filtering Section (SECOND) */}
                  {cfRecommendations.length > 0 && (
                    <div className="recommendation-section">
                      <div className="section-header">
                        <h3>🤝 Collaborative Filtering</h3>
                        <p>Popular items in your categories</p>
                      </div>
                      <ProductGrid
                        products={cfRecommendations}
                        onInteraction={handleInteraction}
                        interactionStates={interactionStates}
                      />
                    </div>
                  )}

                  {/* Empty state */}
                  {cfRecommendations.length === 0 && contentRecommendations.length === 0 && (
                    <div className="empty-recommendations">
                      <div className="empty-icon">✨</div>
                      <h3>Start Building Your Profile</h3>
                      <p>Like, save, or interact with products to get personalized recommendations!</p>
                      <button
                        className="btn-primary"
                        onClick={() => setActiveTab('popular')}
                      >
                        Browse Popular Items
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {activeTab === 'chat' && (
            <Chat
              userId={userId}
              onInteraction={handleInteraction}
              interactionStates={interactionStates}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
