-- =============================================================================
-- ProxiPrep - Self-Healing SQLite Schema
-- Kitchen Intelligence Database Architecture
-- =============================================================================
-- 
-- Design Principles:
--   1. Third Normal Form (3NF) - No transitive dependencies
--   2. Foreign Key enforcement for referential integrity
--   3. Strategic indexing on high-frequency query columns
--   4. Self-healing via UPSERT patterns (INSERT OR REPLACE)
--   5. Audit trails with created_at/updated_at timestamps
--
-- Author: ProxiPrep Engineering
-- Version: 1.0.0
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;  -- Write-Ahead Logging for concurrent reads
PRAGMA synchronous = NORMAL;

-- =============================================================================
-- TABLE 1: categories
-- Lookup table for ingredient/item categories
-- =============================================================================
CREATE TABLE IF NOT EXISTS categories (
    category_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    display_order   INTEGER DEFAULT 0,
    icon            TEXT,  -- Emoji or icon reference
    requires_refrigeration BOOLEAN DEFAULT FALSE,
    is_protein      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed default categories (matches IngredientType enum)
INSERT OR IGNORE INTO categories (name, description, icon, requires_refrigeration, is_protein, display_order) VALUES
    ('meat', 'Beef, Pork, Lamb, Veal', '🥩', TRUE, TRUE, 1),
    ('poultry', 'Chicken, Turkey, Duck', '🍗', TRUE, TRUE, 2),
    ('seafood', 'Fish, Shellfish, Crustaceans', '🐟', TRUE, TRUE, 3),
    ('produce', 'Fruits, Vegetables, Herbs', '🥬', TRUE, FALSE, 4),
    ('sauces', 'Sauces, Dressings, Condiments', '🫙', TRUE, FALSE, 5),
    ('appetizers', 'Starters, Small Plates, Sides', '🍽️', FALSE, FALSE, 6),
    ('bulk_prep', 'Stocks, Batches, Mise en Place', '🍲', TRUE, FALSE, 7),
    ('desserts', 'Pastry, Sweets, Ice Cream', '🍰', TRUE, FALSE, 8),
    ('urgent_86', '86''d items, Critical Shortages', '🚨', FALSE, FALSE, 9),
    ('specials', 'Daily Specials, Limited Items', '⭐', FALSE, FALSE, 10),
    ('other', 'Unclassified Items', '📦', FALSE, FALSE, 99);

CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);


-- =============================================================================
-- TABLE 2: stations
-- Kitchen stations for workflow organization
-- =============================================================================
CREATE TABLE IF NOT EXISTS stations (
    station_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    display_order   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed default stations (matches Station enum)
INSERT OR IGNORE INTO stations (name, description, display_order) VALUES
    ('pantry', 'Pantry (Cold) - Garde Manger, salads, cold apps', 1),
    ('saute', 'Sauté Station - pan work, pastas, risottos', 2),
    ('grill', 'Grill / Broil - steaks, burgers, char items', 3),
    ('fryer', 'Fry Station - fried items, tempura', 4),
    ('prep', 'Bulk Prep - stocks, primals, large batch work', 5),
    ('bake', 'Bakeshop - pastry, bread, desserts', 6),
    ('dry_storage', 'Dry Storage - shelf-stable items', 7),
    ('other', 'Unassigned', 99);

CREATE INDEX IF NOT EXISTS idx_stations_name ON stations(name);


-- =============================================================================
-- TABLE 3: units
-- Standardized units of measurement
-- =============================================================================
CREATE TABLE IF NOT EXISTS units (
    unit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    abbreviation    TEXT NOT NULL UNIQUE,  -- Standard form: lb, oz, pan, etc.
    full_name       TEXT NOT NULL,
    unit_type       TEXT CHECK(unit_type IN ('weight', 'volume', 'count', 'container')),
    conversion_factor REAL,  -- For future unit conversion
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed common kitchen units
INSERT OR IGNORE INTO units (abbreviation, full_name, unit_type) VALUES
    ('lb', 'pound', 'weight'),
    ('oz', 'ounce', 'weight'),
    ('kg', 'kilogram', 'weight'),
    ('g', 'gram', 'weight'),
    ('gal', 'gallon', 'volume'),
    ('qt', 'quart', 'volume'),
    ('pt', 'pint', 'volume'),
    ('cup', 'cup', 'volume'),
    ('ea', 'each', 'count'),
    ('dz', 'dozen', 'count'),
    ('cs', 'case', 'container'),
    ('pan', 'pan', 'container'),
    ('tray', 'tray', 'container'),
    ('bag', 'bag', 'container'),
    ('bunch', 'bunch', 'count'),
    ('head', 'head', 'count'),
    ('can', 'can', 'container'),
    ('bottle', 'bottle', 'container'),
    ('jar', 'jar', 'container');

CREATE INDEX IF NOT EXISTS idx_units_abbreviation ON units(abbreviation);


-- =============================================================================
-- TABLE 4: inventory_items
-- Master table of standardized inventory items (canonical names)
-- =============================================================================
CREATE TABLE IF NOT EXISTS inventory_items (
    item_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL UNIQUE,  -- Standardized name: "Chicken Breast"
    category_id     INTEGER REFERENCES categories(category_id) ON DELETE SET NULL,
    default_station_id INTEGER REFERENCES stations(station_id) ON DELETE SET NULL,
    default_unit_id INTEGER REFERENCES units(unit_id) ON DELETE SET NULL,
    par_level       REAL,  -- Default par level if applicable
    shelf_life_days INTEGER,  -- Days until expiration
    storage_notes   TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inventory_canonical ON inventory_items(canonical_name);
CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory_items(category_id);
CREATE INDEX IF NOT EXISTS idx_inventory_station ON inventory_items(default_station_id);


-- =============================================================================
-- TABLE 5: shorthand_mappings
-- The "Kitchen Thesaurus" - Maps OCR strings to canonical items
-- This is the SELF-HEALING table that learns new abbreviations
-- =============================================================================
CREATE TABLE IF NOT EXISTS shorthand_mappings (
    mapping_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    shorthand       TEXT NOT NULL,  -- Raw OCR text: "Chk", "Chkn", "Chix"
    item_id         INTEGER NOT NULL REFERENCES inventory_items(item_id) ON DELETE CASCADE,
    confidence      REAL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
    source          TEXT CHECK(source IN ('factory', 'user', 'fuzzy', 'learned')),
    times_seen      INTEGER DEFAULT 1,  -- Usage frequency for ranking
    last_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite unique constraint: one shorthand maps to one item
    UNIQUE(shorthand, item_id)
);

-- Critical index for fast shorthand lookups (the hot path)
CREATE INDEX IF NOT EXISTS idx_shorthand_lookup ON shorthand_mappings(shorthand COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_shorthand_item ON shorthand_mappings(item_id);
CREATE INDEX IF NOT EXISTS idx_shorthand_frequency ON shorthand_mappings(times_seen DESC);


-- =============================================================================
-- TABLE 6: unit_aliases
-- Maps unit variations to standard units (similar to shorthand_mappings)
-- =============================================================================
CREATE TABLE IF NOT EXISTS unit_aliases (
    alias_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alias           TEXT NOT NULL,  -- Raw form: "lbs", "pound", "#"
    unit_id         INTEGER NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(alias)
);

-- Seed common aliases
INSERT OR IGNORE INTO unit_aliases (alias, unit_id) VALUES
    ('lbs', (SELECT unit_id FROM units WHERE abbreviation = 'lb')),
    ('pound', (SELECT unit_id FROM units WHERE abbreviation = 'lb')),
    ('pounds', (SELECT unit_id FROM units WHERE abbreviation = 'lb')),
    ('#', (SELECT unit_id FROM units WHERE abbreviation = 'lb')),
    ('ozs', (SELECT unit_id FROM units WHERE abbreviation = 'oz')),
    ('ounce', (SELECT unit_id FROM units WHERE abbreviation = 'oz')),
    ('ounces', (SELECT unit_id FROM units WHERE abbreviation = 'oz')),
    ('each', (SELECT unit_id FROM units WHERE abbreviation = 'ea')),
    ('pcs', (SELECT unit_id FROM units WHERE abbreviation = 'ea')),
    ('pc', (SELECT unit_id FROM units WHERE abbreviation = 'ea')),
    ('qts', (SELECT unit_id FROM units WHERE abbreviation = 'qt')),
    ('quart', (SELECT unit_id FROM units WHERE abbreviation = 'qt')),
    ('quarts', (SELECT unit_id FROM units WHERE abbreviation = 'qt')),
    ('gals', (SELECT unit_id FROM units WHERE abbreviation = 'gal')),
    ('gallon', (SELECT unit_id FROM units WHERE abbreviation = 'gal')),
    ('gallons', (SELECT unit_id FROM units WHERE abbreviation = 'gal')),
    ('pans', (SELECT unit_id FROM units WHERE abbreviation = 'pan')),
    ('trays', (SELECT unit_id FROM units WHERE abbreviation = 'tray')),
    ('bags', (SELECT unit_id FROM units WHERE abbreviation = 'bag')),
    ('cases', (SELECT unit_id FROM units WHERE abbreviation = 'cs')),
    ('case', (SELECT unit_id FROM units WHERE abbreviation = 'cs')),
    ('box', (SELECT unit_id FROM units WHERE abbreviation = 'cs')),
    ('boxes', (SELECT unit_id FROM units WHERE abbreviation = 'cs')),
    ('doz', (SELECT unit_id FROM units WHERE abbreviation = 'dz')),
    ('dozen', (SELECT unit_id FROM units WHERE abbreviation = 'dz'));

CREATE INDEX IF NOT EXISTS idx_unit_aliases ON unit_aliases(alias COLLATE NOCASE);


-- =============================================================================
-- TABLE 7: prep_sheets
-- Metadata for each scanned prep sheet (the "header" record)
-- =============================================================================
CREATE TABLE IF NOT EXISTS prep_sheets (
    sheet_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path      TEXT,  -- Original image file path
    image_hash      TEXT,  -- SHA-256 for deduplication
    ocr_engine      TEXT,  -- 'openai_gpt4o', 'azure_di', 'google_vision', etc.
    extraction_mode TEXT,  -- 'single_pass', 'comprehensive', 'ensemble'
    total_items     INTEGER DEFAULT 0,
    verified_count  INTEGER DEFAULT 0,
    review_count    INTEGER DEFAULT 0,
    ignored_count   INTEGER DEFAULT 0,
    confidence_avg  REAL,  -- Average OCR confidence
    processing_time_ms INTEGER,
    notes           TEXT,
    created_by      TEXT,  -- User/device identifier
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prep_sheets_timestamp ON prep_sheets(scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prep_sheets_hash ON prep_sheets(image_hash);


-- =============================================================================
-- TABLE 8: line_items
-- Individual items extracted from prep sheets
-- =============================================================================
CREATE TABLE IF NOT EXISTS line_items (
    line_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sheet_id        INTEGER NOT NULL REFERENCES prep_sheets(sheet_id) ON DELETE CASCADE,
    item_id         INTEGER REFERENCES inventory_items(item_id) ON DELETE SET NULL,
    
    -- Raw OCR data (preserved for auditing)
    raw_text        TEXT NOT NULL,
    raw_quantity    REAL,
    raw_unit        TEXT,
    
    -- Resolved/normalized data
    resolved_name   TEXT,
    quantity        REAL,
    unit_id         INTEGER REFERENCES units(unit_id) ON DELETE SET NULL,
    
    -- Par check support (count_on_hand / par_level)
    is_par_check    BOOLEAN DEFAULT FALSE,
    count_on_hand   REAL,
    par_level       REAL,
    
    -- Confidence & status
    ocr_confidence  REAL CHECK(ocr_confidence >= 0 AND ocr_confidence <= 1),
    resolution_source TEXT CHECK(resolution_source IN ('thesaurus', 'fuzzy_match', 'user_defined', 'raw')),
    status          TEXT CHECK(status IN ('verified', 'auto_corrected', 'needs_review', 'ignored')),
    
    -- Position on the sheet (for ordering)
    position        INTEGER DEFAULT 0,
    
    -- Flags
    is_urgent       BOOLEAN DEFAULT FALSE,
    
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_line_items_sheet ON line_items(sheet_id);
CREATE INDEX IF NOT EXISTS idx_line_items_item ON line_items(item_id);
CREATE INDEX IF NOT EXISTS idx_line_items_status ON line_items(status);
CREATE INDEX IF NOT EXISTS idx_line_items_timestamp ON line_items(created_at DESC);


-- =============================================================================
-- TABLE 9: historical_usage
-- Aggregated daily usage data for forecasting
-- =============================================================================
CREATE TABLE IF NOT EXISTS historical_usage (
    usage_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         INTEGER NOT NULL REFERENCES inventory_items(item_id) ON DELETE CASCADE,
    usage_date      DATE NOT NULL,
    quantity_used   REAL NOT NULL,
    unit_id         INTEGER REFERENCES units(unit_id),
    day_of_week     INTEGER CHECK(day_of_week >= 0 AND day_of_week <= 6),  -- 0=Monday
    is_holiday      BOOLEAN DEFAULT FALSE,
    weather_condition TEXT,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- One record per item per day
    UNIQUE(item_id, usage_date)
);

CREATE INDEX IF NOT EXISTS idx_historical_item ON historical_usage(item_id);
CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_usage(usage_date DESC);
CREATE INDEX IF NOT EXISTS idx_historical_dow ON historical_usage(day_of_week);


-- =============================================================================
-- TABLE 10: forecasts
-- Predicted prep amounts from the forecasting model
-- =============================================================================
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         INTEGER NOT NULL REFERENCES inventory_items(item_id) ON DELETE CASCADE,
    forecast_date   DATE NOT NULL,
    predicted_qty   REAL NOT NULL,
    confidence_lower REAL,  -- Lower bound of prediction interval
    confidence_upper REAL,  -- Upper bound of prediction interval
    model_version   TEXT,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(item_id, forecast_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_item ON forecasts(item_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(forecast_date);


-- =============================================================================
-- TABLE 11: user_corrections
-- Audit log of user corrections for active learning
-- =============================================================================
CREATE TABLE IF NOT EXISTS user_corrections (
    correction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    line_item_id    INTEGER REFERENCES line_items(line_id) ON DELETE SET NULL,
    original_text   TEXT NOT NULL,
    corrected_to    TEXT NOT NULL,
    category_id     INTEGER REFERENCES categories(category_id),
    added_to_thesaurus BOOLEAN DEFAULT FALSE,
    corrected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    corrected_by    TEXT
);

CREATE INDEX IF NOT EXISTS idx_corrections_timestamp ON user_corrections(corrected_at DESC);


-- =============================================================================
-- VIEWS: Convenient pre-joined queries
-- =============================================================================

-- View: Full line item details with resolved names
CREATE VIEW IF NOT EXISTS v_line_items_full AS
SELECT 
    li.line_id,
    li.sheet_id,
    ps.scan_timestamp,
    li.raw_text,
    li.raw_quantity,
    li.raw_unit,
    COALESCE(li.resolved_name, ii.canonical_name, li.raw_text) AS display_name,
    li.quantity,
    u.abbreviation AS unit,
    c.name AS category,
    c.icon AS category_icon,
    s.name AS station,
    li.ocr_confidence,
    li.resolution_source,
    li.status,
    li.is_urgent,
    li.is_par_check,
    li.count_on_hand,
    li.par_level
FROM line_items li
LEFT JOIN prep_sheets ps ON li.sheet_id = ps.sheet_id
LEFT JOIN inventory_items ii ON li.item_id = ii.item_id
LEFT JOIN units u ON li.unit_id = u.unit_id
LEFT JOIN categories c ON ii.category_id = c.category_id
LEFT JOIN stations s ON ii.default_station_id = s.station_id;


-- View: Shorthand lookup with item details
CREATE VIEW IF NOT EXISTS v_shorthand_lookup AS
SELECT 
    sm.shorthand,
    ii.canonical_name,
    ii.item_id,
    c.name AS category,
    sm.confidence,
    sm.source,
    sm.times_seen
FROM shorthand_mappings sm
JOIN inventory_items ii ON sm.item_id = ii.item_id
LEFT JOIN categories c ON ii.category_id = c.category_id
ORDER BY sm.times_seen DESC;


-- View: Daily usage summary for forecasting
CREATE VIEW IF NOT EXISTS v_daily_usage_summary AS
SELECT 
    ii.canonical_name,
    hu.usage_date,
    hu.quantity_used,
    u.abbreviation AS unit,
    hu.day_of_week,
    CASE hu.day_of_week
        WHEN 0 THEN 'Monday'
        WHEN 1 THEN 'Tuesday'
        WHEN 2 THEN 'Wednesday'
        WHEN 3 THEN 'Thursday'
        WHEN 4 THEN 'Friday'
        WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END AS day_name,
    hu.is_holiday
FROM historical_usage hu
JOIN inventory_items ii ON hu.item_id = ii.item_id
LEFT JOIN units u ON hu.unit_id = u.unit_id
ORDER BY hu.usage_date DESC;


-- =============================================================================
-- TRIGGERS: Auto-update timestamps and maintain integrity
-- =============================================================================

-- Trigger: Update timestamp on inventory_items modification
CREATE TRIGGER IF NOT EXISTS trg_inventory_items_updated
AFTER UPDATE ON inventory_items
BEGIN
    UPDATE inventory_items SET updated_at = CURRENT_TIMESTAMP WHERE item_id = NEW.item_id;
END;

-- Trigger: Update timestamp on line_items modification
CREATE TRIGGER IF NOT EXISTS trg_line_items_updated
AFTER UPDATE ON line_items
BEGIN
    UPDATE line_items SET updated_at = CURRENT_TIMESTAMP WHERE line_id = NEW.line_id;
END;

-- Trigger: Update shorthand usage stats when mapping is used
CREATE TRIGGER IF NOT EXISTS trg_shorthand_usage
AFTER INSERT ON line_items
WHEN NEW.item_id IS NOT NULL
BEGIN
    UPDATE shorthand_mappings 
    SET times_seen = times_seen + 1, last_seen = CURRENT_TIMESTAMP
    WHERE item_id = NEW.item_id 
    AND shorthand = NEW.raw_text COLLATE NOCASE;
END;

-- Trigger: Update prep_sheet counts after line_item insert
CREATE TRIGGER IF NOT EXISTS trg_update_sheet_counts_insert
AFTER INSERT ON line_items
BEGIN
    UPDATE prep_sheets SET
        total_items = (SELECT COUNT(*) FROM line_items WHERE sheet_id = NEW.sheet_id),
        verified_count = (SELECT COUNT(*) FROM line_items WHERE sheet_id = NEW.sheet_id AND status = 'verified'),
        review_count = (SELECT COUNT(*) FROM line_items WHERE sheet_id = NEW.sheet_id AND status = 'needs_review'),
        ignored_count = (SELECT COUNT(*) FROM line_items WHERE sheet_id = NEW.sheet_id AND status = 'ignored')
    WHERE sheet_id = NEW.sheet_id;
END;


-- =============================================================================
-- SCHEMA VERSION TRACKING
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_version (
    version_id      INTEGER PRIMARY KEY,
    version_number  TEXT NOT NULL,
    applied_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description     TEXT
);

INSERT OR IGNORE INTO schema_version (version_id, version_number, description) 
VALUES (1, '1.0.0', 'Initial schema with full normalization and self-healing support');
