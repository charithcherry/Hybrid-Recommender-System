/**
 * User Login/Registration Component
 */

import React, { useState } from 'react';
import { registerUser, loginUser } from '../services/api';
import './UserLogin.css';

const UserLogin = ({ onLoginSuccess }) => {
  const [mode, setMode] = useState('login'); // 'login' or 'register'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!username.trim()) {
      setError('Please enter a username');
      return;
    }

    if (!password.trim()) {
      setError('Please enter a password');
      return;
    }

    if (password.length < 4) {
      setError('Password must be at least 4 characters');
      return;
    }

    setLoading(true);
    setError('');

    try {
      if (mode === 'register') {
        // Register new user
        const userData = await registerUser(username.trim(), password);
        console.log('User registered:', userData);

        // Store in localStorage
        localStorage.setItem('userId', userData.user_id);
        localStorage.setItem('username', userData.username);

        // Notify parent component
        onLoginSuccess(userData);
      } else {
        // Login existing user
        const response = await loginUser(username.trim(), password);

        if (response.success) {
          console.log('User logged in:', response);

          // Store in localStorage
          localStorage.setItem('userId', response.user_id);
          localStorage.setItem('username', response.username);

          // Notify parent with user data
          onLoginSuccess({
            user_id: response.user_id,
            username: response.username
          });
        } else {
          setError(response.message);
        }
      }
    } catch (err) {
      console.error('Auth error:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError(mode === 'register' ? 'Failed to register. Please try again.' : 'Failed to login. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setMode(mode === 'login' ? 'register' : 'login');
    setError('');
    setPassword('');
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Fashion Recommender</h1>
        <p>{mode === 'login' ? 'Welcome back!' : 'Create your account'}</p>

        <div className="mode-tabs">
          <button
            className={`tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => setMode('login')}
            type="button"
          >
            Login
          </button>
          <button
            className={`tab ${mode === 'register' ? 'active' : ''}`}
            onClick={() => setMode('register')}
            type="button"
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              disabled={loading}
              maxLength={50}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password (min 4 characters)"
              disabled={loading}
              minLength={4}
              required
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" disabled={loading} className="login-button">
            {loading ? (mode === 'register' ? 'Registering...' : 'Logging in...') : (mode === 'register' ? 'Create Account' : 'Login')}
          </button>
        </form>

        <div className="info-text">
          <p>
            {mode === 'login'
              ? "Don't have an account? Click 'Register' above."
              : 'Get personalized recommendations as you browse!'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default UserLogin;
