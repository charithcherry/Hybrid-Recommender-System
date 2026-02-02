/**
 * Product Card Component - Displays individual product with interaction buttons
 * Features: Facebook-style toggle behavior, interaction state tracking
 */

import React, { useState, useEffect } from 'react';
import { FaHeart, FaHeartBroken, FaBookmark, FaShoppingCart } from 'react-icons/fa';
import './ProductCard.css';

const ProductCard = ({ product, onInteraction, interactionStates = {} }) => {
  const [loading, setLoading] = useState(false);
  const [currentInteraction, setCurrentInteraction] = useState(null);

  // Update current interaction from interactionStates prop
  useEffect(() => {
    const itemInteraction = interactionStates[product.item_id];
    setCurrentInteraction(itemInteraction || null);
  }, [interactionStates, product.item_id]);

  const handleInteraction = async (type) => {
    // Check if clicking the same button (toggle off)
    if (currentInteraction === type) {
      // Toggle off - remove interaction
      setLoading(true);
      try {
        await onInteraction(product.item_id, type);
        setCurrentInteraction(null);
      } catch (error) {
        console.error('Interaction error:', error);
      }
      setLoading(false);
    } else {
      // Add or switch interaction
      setLoading(true);
      try {
        await onInteraction(product.item_id, type);
        setCurrentInteraction(type);
      } catch (error) {
        console.error('Interaction error:', error);
      }
      setLoading(false);
    }
  };

  // Fixed-size placeholder image (300x300)
  const placeholderImage = `https://via.placeholder.com/300x300/e8e8e8/666666?text=${encodeURIComponent('Fashion')}`;

  // Clean product title
  const displayTitle = product.title && product.title.trim() ? product.title : `Fashion Item #${product.item_id}`;

  return (
    <div className="product-card">
      <div className="product-image-container">
        <img
          src={placeholderImage}
          alt={displayTitle}
          className="product-image"
          loading="lazy"
        />
        {product.score !== undefined && (
          <div className="score-badge">
            {product.score.toFixed(1)}
            {product.interaction_count !== undefined && product.interaction_count !== null && (
              <span className="interaction-count-badge">
                 👥 {product.interaction_count}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="product-info">
        <h3 className="product-title">
          {displayTitle}
        </h3>

        <div className="product-meta">
          <span className="item-id">#{product.item_id}</span>
        </div>
      </div>

      <div className="interaction-buttons">
        <button
          className={`btn-interaction ${currentInteraction === 'like' ? 'active' : ''} ${loading ? 'loading' : ''}`}
          onClick={() => handleInteraction('like')}
          disabled={loading}
          title={currentInteraction === 'like' ? 'Unlike' : 'Like'}
        >
          <FaHeart />
        </button>

        <button
          className={`btn-interaction ${currentInteraction === 'dislike' ? 'active' : ''} ${loading ? 'loading' : ''}`}
          onClick={() => handleInteraction('dislike')}
          disabled={loading}
          title={currentInteraction === 'dislike' ? 'Remove dislike' : 'Dislike'}
        >
          <FaHeartBroken />
        </button>

        <button
          className={`btn-interaction ${currentInteraction === 'save' ? 'active' : ''} ${loading ? 'loading' : ''}`}
          onClick={() => handleInteraction('save')}
          disabled={loading}
          title={currentInteraction === 'save' ? 'Unsave' : 'Save'}
        >
          <FaBookmark />
        </button>

        <button
          className={`btn-interaction btn-buy ${currentInteraction === 'buy' ? 'active' : ''} ${loading ? 'loading' : ''}`}
          onClick={() => handleInteraction('buy')}
          disabled={loading}
          title={currentInteraction === 'buy' ? 'Remove from cart' : 'Buy'}
        >
          <FaShoppingCart />
        </button>
      </div>
    </div>
  );
};

export default ProductCard;
