/**
 * Chat Component - Natural Language Product Search
 * Allows users to search for products using conversational queries
 */

import React, { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../services/api';
import ProductCard from './ProductCard';
import './Chat.css';

function Chat({ userId, onInteraction, interactionStates }) {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;

    const userMessage = inputValue.trim();
    setInputValue('');

    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toLocaleTimeString()
    }]);

    setLoading(true);

    try {
      // Call chat API
      const response = await sendChatMessage(
        userMessage,
        userId,
        conversationId,
        10
      );

      // Update conversation ID
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add bot response to chat
      setMessages(prev => [...prev, {
        type: 'bot',
        content: response.explanation,
        intent: response.understood_intent,
        products: response.recommendations,
        timestamp: new Date().toLocaleTimeString(),
        retrievalTime: response.retrieval_time_ms
      }]);

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const exampleQueries = [
    "Show me red dresses for a summer wedding",
    "I need casual watches for men under $100",
    "What backpacks would go well with a college style?",
    "Find me black formal shoes",
    "Show me winter jackets for women"
  ];

  const handleExampleClick = (query) => {
    setInputValue(query);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>💬 Ask for Recommendations</h2>
        <p className="chat-subtitle">
          Describe what you're looking for in natural language
        </p>
      </div>

      {messages.length === 0 && (
        <div className="chat-welcome">
          <div className="welcome-icon">🤖</div>
          <h3>Hi! I'm your fashion assistant</h3>
          <p>Try asking me something like:</p>
          <div className="example-queries">
            {exampleQueries.map((query, idx) => (
              <button
                key={idx}
                className="example-query"
                onClick={() => handleExampleClick(query)}
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-messages">
        {messages.map((message, idx) => (
          <div key={idx} className={`message message-${message.type}`}>
            {message.type === 'user' && (
              <div className="message-bubble user-message">
                <div className="message-content">{message.content}</div>
                <div className="message-time">{message.timestamp}</div>
              </div>
            )}

            {message.type === 'bot' && (
              <div className="bot-response">
                <div className="message-bubble bot-message">
                  <div className="message-content">{message.content}</div>
                  {message.intent && (
                    <div className="intent-display">
                      <strong>Understood:</strong>{' '}
                      {Object.entries(message.intent.filters || {}).map(([key, value]) => (
                        <span key={key} className="filter-tag">
                          {key}: {value}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="message-time">
                    {message.timestamp} • {message.retrievalTime.toFixed(0)}ms
                  </div>
                </div>

                {message.products && message.products.length > 0 && (
                  <div className="chat-products">
                    <div className="products-grid">
                      {message.products.map((product) => (
                        <div key={product.item_id} className="chat-product-card">
                          <ProductCard
                            product={{
                              id: product.item_id,
                              productDisplayName: product.title,
                              ...product.metadata
                            }}
                            score={product.score}
                            onInteraction={onInteraction}
                            interactionState={interactionStates[product.item_id]}
                          />
                          {product.explanation && (
                            <div className="product-explanation">
                              {product.explanation}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {message.type === 'error' && (
              <div className="message-bubble error-message">
                <div className="message-content">❌ {message.content}</div>
                <div className="message-time">{message.timestamp}</div>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message message-bot">
            <div className="message-bubble bot-message loading">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          className="chat-input"
          placeholder="Ask me anything... (e.g., 'Show me red dresses for summer')"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          className="chat-send-button"
          disabled={!inputValue.trim() || loading}
        >
          {loading ? '⏳' : '➤'}
        </button>
      </form>
    </div>
  );
}

export default Chat;
