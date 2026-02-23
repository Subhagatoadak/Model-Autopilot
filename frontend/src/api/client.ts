import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

// Interceptor to add OpenAI API key from localStorage
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('openai_api_key');
  if (apiKey) {
    config.headers['X-OpenAI-API-Key'] = apiKey;
  }
  return config;
});

export default api;
