/**
 * Product Grid Component - Pinterest-style masonry layout
 */

import React from 'react';
import Masonry from 'react-masonry-css';
import ProductCard from './ProductCard';
import './ProductGrid.css';

const ProductGrid = ({ products, onInteraction, loading, interactionStates = {} }) => {
  const breakpointColumns = {
    default: 4,
    1400: 3,
    1000: 2,
    600: 1
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading products...</p>
      </div>
    );
  }

  if (!products || products.length === 0) {
    return (
      <div className="empty-state">
        <h2>No products found</h2>
        <p>Click "Get Recommendations" to see personalized items!</p>
      </div>
    );
  }

  return (
    <Masonry
      breakpointCols={breakpointColumns}
      className="product-grid"
      columnClassName="product-grid-column"
    >
      {products.map((product) => (
        <ProductCard
          key={product.item_id}
          product={product}
          onInteraction={onInteraction}
          interactionStates={interactionStates}
        />
      ))}
    </Masonry>
  );
};

export default ProductGrid;
