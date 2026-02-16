import React from 'react';
import './Sidebar.css';

function Sidebar({ activeTab, onTabChange, user, interactionCount, onLogout }) {
  const tabs = [
    { id: 'all', icon: '📱', label: 'All Products' },
    { id: 'popular', icon: '🔥', label: 'Top 50' },
    { id: 'foryou', icon: '✨', label: 'For You' },
    { id: 'chat', icon: '💬', label: 'Chat' }
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-content">
        {/* Navigation Tabs */}
        <div className="sidebar-nav">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`sidebar-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span className="tab-label">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* User Profile Section */}
        <div className="sidebar-footer">
          <div className="sidebar-divider"></div>

          <div className="user-profile-section">
            <div className="user-avatar">
              <span className="avatar-icon">👤</span>
            </div>
            <div className="user-info">
              <div className="user-name">{user?.username || 'User'}</div>
              <div className="user-stats">
                <span className="interaction-badge">
                  {interactionCount} interaction{interactionCount !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
          </div>

          <button className="logout-btn" onClick={onLogout}>
            <span className="logout-icon">🚪</span>
            <span>Logout</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
