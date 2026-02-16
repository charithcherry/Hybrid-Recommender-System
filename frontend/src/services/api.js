/**
 * API Service - Handles all communication with FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';
const CHAT_API_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const chatApi = axios.create({
  baseURL: CHAT_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// User Management
export const registerUser = async (username, password, email = null) => {
  const response = await api.post('/users/register', { username, password, email });
  return response.data;
};

export const loginUser = async (username, password) => {
  const response = await api.post('/users/login', { username, password });
  return response.data;
};

export const getUserProfile = async (userId) => {
  const response = await api.get(`/users/${userId}`);
  return response.data;
};

export const getUserInteractions = async (userId) => {
  const response = await api.get(`/users/${userId}/interactions`);
  return response.data;
};

// Interaction Tracking
export const trackInteraction = async (userId, itemId, interactionType) => {
  const response = await api.post('/interactions', {
    user_id: userId,
    item_id: itemId,
    interaction_type: interactionType,
  });
  return response.data;
};

// Recommendations
export const getColdStartRecommendations = async (n = 50, filters = {}) => {
  const params = new URLSearchParams({ n: n.toString() });

  // Add filters if provided
  if (filters.gender) params.append('gender', filters.gender);
  if (filters.masterCategory) params.append('masterCategory', filters.masterCategory);
  if (filters.subCategory) params.append('subCategory', filters.subCategory);
  if (filters.articleType) params.append('articleType', filters.articleType);
  if (filters.baseColour) params.append('baseColour', filters.baseColour);
  if (filters.season) params.append('season', filters.season);
  if (filters.usage) params.append('usage', filters.usage);

  const response = await api.get(`/recommend/coldstart?${params.toString()}`);
  return response.data;
};

export const getPersonalizedRecommendations = async (userId, n = 20) => {
  const response = await api.get(`/recommend/${userId}?n=${n}`);
  return response.data;
};

export const getSplitRecommendations = async (userId, n = 10, filters = {}) => {
  const params = new URLSearchParams({ n: n.toString() });

  // Add filters if provided
  if (filters.gender) params.append('gender', filters.gender);
  if (filters.masterCategory) params.append('masterCategory', filters.masterCategory);
  if (filters.subCategory) params.append('subCategory', filters.subCategory);
  if (filters.articleType) params.append('articleType', filters.articleType);
  if (filters.baseColour) params.append('baseColour', filters.baseColour);
  if (filters.season) params.append('season', filters.season);
  if (filters.usage) params.append('usage', filters.usage);

  const response = await api.get(`/recommend/${userId}/split?${params.toString()}`);
  return response.data;
};

// Product Information
export const getItemInfo = async (itemId) => {
  const response = await api.get(`/items/${itemId}`);
  return response.data;
};

export const getSimilarItems = async (itemId, n = 5) => {
  const response = await api.get(`/similar/${itemId}?n=${n}`);
  return response.data;
};

export const getAllProducts = async (page = 1, limit = 50, filters = {}) => {
  const params = new URLSearchParams({
    page: page.toString(),
    limit: limit.toString()
  });

  // Add filters if provided
  if (filters.gender) params.append('gender', filters.gender);
  if (filters.masterCategory) params.append('masterCategory', filters.masterCategory);
  if (filters.subCategory) params.append('subCategory', filters.subCategory);
  if (filters.articleType) params.append('articleType', filters.articleType);
  if (filters.baseColour) params.append('baseColour', filters.baseColour);
  if (filters.season) params.append('season', filters.season);
  if (filters.usage) params.append('usage', filters.usage);

  const response = await api.get(`/items/all?${params.toString()}`);
  return response.data;
};

export const getFilterValues = async () => {
  const response = await api.get('/items/filter-values');
  return response.data;
};

export const getUserInteractionStates = async (userId) => {
  const response = await api.get(`/users/${userId}/interaction-states`);
  return response.data;
};

// System Info
export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

// Conversational Chat (port 8001)
export const sendChatMessage = async (query, userId = null, conversationId = null, n = 10) => {
  const response = await chatApi.post('/chat', {
    query,
    user_id: userId,
    conversation_id: conversationId,
    n
  });
  return response.data;
};

export const getConversationHistory = async (conversationId) => {
  const response = await chatApi.get(`/conversations/${conversationId}`);
  return response.data;
};

export default api;
