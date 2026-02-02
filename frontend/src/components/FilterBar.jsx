import React, { useState, useEffect } from 'react';
import { getFilterValues } from '../services/api';
import './FilterBar.css';

function FilterBar({ filters, onFilterChange }) {
  const [filterValues, setFilterValues] = useState({
    genders: [],
    masterCategories: [],
    subCategories: [],
    articleTypes: [],
    baseColours: [],
    seasons: [],
    usages: []
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadFilterValues();
  }, []);

  const loadFilterValues = async () => {
    try {
      const values = await getFilterValues();
      setFilterValues(values);
      setLoading(false);
    } catch (error) {
      console.error('Error loading filter values:', error);
      setLoading(false);
    }
  };

  const handleFilterChange = (filterName, value) => {
    onFilterChange({
      ...filters,
      [filterName]: value === '' ? null : value
    });
  };

  const clearFilters = () => {
    onFilterChange({
      gender: null,
      masterCategory: null,
      subCategory: null,
      articleType: null,
      baseColour: null,
      season: null,
      usage: null
    });
  };

  const hasActiveFilters = Object.values(filters).some(v => v !== null);

  if (loading) {
    return <div className="filter-bar loading">Loading filters...</div>;
  }

  return (
    <div className="filter-bar">
      <div className="filter-controls">
        {/* Gender Filter */}
        <select
          className="filter-select"
          value={filters.gender || ''}
          onChange={(e) => handleFilterChange('gender', e.target.value)}
        >
          <option value="">All Genders</option>
          {filterValues.genders.map(gender => (
            <option key={gender} value={gender}>{gender}</option>
          ))}
        </select>

        {/* Master Category Filter */}
        <select
          className="filter-select"
          value={filters.masterCategory || ''}
          onChange={(e) => handleFilterChange('masterCategory', e.target.value)}
        >
          <option value="">All Categories</option>
          {filterValues.masterCategories.map(cat => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>

        {/* Sub Category Filter */}
        <select
          className="filter-select"
          value={filters.subCategory || ''}
          onChange={(e) => handleFilterChange('subCategory', e.target.value)}
        >
          <option value="">All Sub-Categories</option>
          {filterValues.subCategories.map(cat => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>

        {/* Article Type Filter */}
        <select
          className="filter-select"
          value={filters.articleType || ''}
          onChange={(e) => handleFilterChange('articleType', e.target.value)}
        >
          <option value="">All Types</option>
          {filterValues.articleTypes.map(type => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>

        {/* Color Filter */}
        <select
          className="filter-select"
          value={filters.baseColour || ''}
          onChange={(e) => handleFilterChange('baseColour', e.target.value)}
        >
          <option value="">All Colors</option>
          {filterValues.baseColours.map(color => (
            <option key={color} value={color}>{color}</option>
          ))}
        </select>

        {/* Season Filter */}
        <select
          className="filter-select"
          value={filters.season || ''}
          onChange={(e) => handleFilterChange('season', e.target.value)}
        >
          <option value="">All Seasons</option>
          {filterValues.seasons.map(season => (
            <option key={season} value={season}>{season}</option>
          ))}
        </select>

        {/* Usage Filter */}
        <select
          className="filter-select"
          value={filters.usage || ''}
          onChange={(e) => handleFilterChange('usage', e.target.value)}
        >
          <option value="">All Usage</option>
          {filterValues.usages.map(usage => (
            <option key={usage} value={usage}>{usage}</option>
          ))}
        </select>

        {/* Clear Filters Button */}
        {hasActiveFilters && (
          <button className="clear-filters-btn" onClick={clearFilters}>
            ✕ Clear All
          </button>
        )}
      </div>

      {/* Active Filters Display */}
      {hasActiveFilters && (
        <div className="active-filters">
          <span className="active-filters-label">Active filters:</span>
          {Object.entries(filters).map(([key, value]) => {
            if (value) {
              return (
                <span key={key} className="filter-tag">
                  {value}
                  <button
                    className="filter-tag-remove"
                    onClick={() => handleFilterChange(key, '')}
                  >
                    ✕
                  </button>
                </span>
              );
            }
            return null;
          })}
        </div>
      )}
    </div>
  );
}

export default FilterBar;
