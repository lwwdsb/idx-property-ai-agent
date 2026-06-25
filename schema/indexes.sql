-- Indexes for high-frequency filter columns (Week 0 stage 3).
-- Idempotent-ish: run against idx_exchange. rets_property already ships with most
-- indexes from the dump (L_City/L_Zip/L_Type_/L_DisplayId/L_ListingID + ft_remarks
-- FULLTEXT + PRIMARY id); this only adds what's missing. california_sold ships with none.
-- MySQL has no "CREATE INDEX IF NOT EXISTS", so re-running may error on existing keys — that's expected.

-- rets_property: add status filter index (others already present from dump)
ALTER TABLE rets_property ADD INDEX idx_L_Status (L_Status);

-- california_sold: had zero indexes
ALTER TABLE california_sold ADD INDEX idx_City (City);
ALTER TABLE california_sold ADD INDEX idx_ListingKey (ListingKey);
