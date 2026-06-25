/**
 * Field dictionary — the SINGLE SOURCE OF TRUTH for MLS column names.
 *
 * Why this exists (highest-value piece of the project):
 * The `rets_property` table uses cryptic physical column names (`L_Keyword2` = beds,
 * `LM_Dec_3` = baths, `LM_Int2_3` = sqft, `L_SystemPrice` = price...), while
 * `california_sold` uses clean names (`BedroomsTotal`, `ClosePrice`...). Scattering
 * these magic names across every query/agent is error-prone. Instead, ALL SQL and
 * agents reference a semantic concept (e.g. `col('beds', 'rets_property')`), and the
 * mapping lives here. This dictionary also feeds the Week 8 RAG "field definitions"
 * knowledge source and the Week 12 schema docs — generate, don't duplicate.
 *
 * Column names verified against the real dumps in Week 0.
 */

export type TableName = 'rets_property' | 'california_sold' | 'rets_openhouse';

export type FieldType = 'string' | 'number' | 'date' | 'text' | 'geo' | 'flag';

export interface FieldDef {
  /** Human-readable label — used for RAG answers and generated docs. */
  label: string;
  type: FieldType;
  /** Physical column name per table. Absent = field not present on that table. */
  columns: Partial<Record<TableName, string>>;
  /** Caveats worth knowing (data quality, units, encoding). */
  note?: string;
}

/**
 * Canonical semantic fields. Add new concepts here; never hard-code a physical
 * column name anywhere else in the codebase.
 */
export const FIELDS = {
  // ---- Identity ----
  listingId:   { label: 'Listing / System ID', type: 'string', columns: { rets_property: 'L_ListingID', rets_openhouse: 'L_ListingID' } },
  listingKey:  { label: 'Listing Key (sold)',  type: 'number', columns: { california_sold: 'ListingKey' }, note: 'Join: CAST(rets_property.L_ListingID AS UNSIGNED) = california_sold.ListingKey' },
  mlsNumber:   { label: 'MLS #',               type: 'string', columns: { rets_property: 'L_DisplayId', rets_openhouse: 'L_DisplayId' } },

  // ---- Location ----
  address:     { label: 'Address',     type: 'string', columns: { rets_property: 'L_Address', california_sold: 'UnparsedAddress' } },
  city:        { label: 'City',        type: 'string', columns: { rets_property: 'L_City', california_sold: 'City' } },
  state:       { label: 'State',       type: 'string', columns: { rets_property: 'L_State', california_sold: 'StateOrProvince' } },
  zip:         { label: 'ZIP / Postal Code', type: 'string', columns: { rets_property: 'L_Zip', california_sold: 'PostalCode' } },
  subdivision: { label: 'Subdivision', type: 'string', columns: { rets_property: 'SubdivisionName', california_sold: 'SubdivisionName' } },
  latitude:    { label: 'Latitude',    type: 'geo',    columns: { rets_property: 'LMD_MP_Latitude', california_sold: 'Latitude' } },
  longitude:   { label: 'Longitude',   type: 'geo',    columns: { rets_property: 'LMD_MP_Longitude', california_sold: 'Longitude' } },

  // ---- Property attributes ----
  propertyType:    { label: 'Property Type',     type: 'string', columns: { rets_property: 'L_Type_', california_sold: 'PropertyType' } },
  propertySubType: { label: 'Property Sub-Type', type: 'string', columns: { california_sold: 'PropertySubType' } },
  beds:        { label: 'Bedrooms',  type: 'number', columns: { rets_property: 'L_Keyword2', california_sold: 'BedroomsTotal' } },
  baths:       { label: 'Bathrooms', type: 'number', columns: { rets_property: 'LM_Dec_3', california_sold: 'BathroomsTotalInteger' } },
  livingArea:  { label: 'Living Area (sqft)', type: 'number', columns: { rets_property: 'LM_Int2_3', california_sold: 'LivingArea' } },
  yearBuilt:   { label: 'Year Built', type: 'number', columns: { rets_property: 'YearBuilt', california_sold: 'YearBuilt' } },
  pool:        { label: 'Private Pool', type: 'flag', columns: { rets_property: 'PoolPrivateYN', california_sold: 'PoolPrivateYN' }, note: 'Stored as varchar Y/N-ish; normalize before comparing.' },
  view:        { label: 'View',         type: 'flag', columns: { rets_property: 'ViewYN', california_sold: 'ViewYN' } },
  fireplace:   { label: 'Fireplace',    type: 'flag', columns: { rets_property: 'FireplaceYN', california_sold: 'FireplaceYN' } },
  hoaFee:      { label: 'HOA Fee',      type: 'number', columns: { rets_property: 'AssociationFee', california_sold: 'AssociationFee' } },

  // ---- Pricing ----
  price:             { label: 'List Price',          type: 'number', columns: { rets_property: 'L_SystemPrice', california_sold: 'ListPrice' } },
  originalListPrice: { label: 'Original List Price', type: 'number', columns: { california_sold: 'OriginalListPrice' } },
  closePrice:        { label: 'Sold / Close Price',  type: 'number', columns: { california_sold: 'ClosePrice' } },

  // ---- Status / dates / market ----
  status:       { label: 'Listing Status', type: 'string', columns: { rets_property: 'L_Status' }, note: 'rets_property is all Active in current dump.' },
  daysOnMarket: { label: 'Days on Market', type: 'number', columns: { rets_property: 'DaysOnMarket', california_sold: 'DaysOnMarket' } },
  listingDate:  { label: 'Listing Date',   type: 'date',   columns: { rets_property: 'ListingContractDate', california_sold: 'ListingContractDate' } },
  closeDate:    { label: 'Close Date',      type: 'date',   columns: { california_sold: 'CloseDate' }, note: 'Stored as VARCHAR with dirty values (e.g. year 2072) — clean/parse before use (Week 5).' },
  modifiedAt:   { label: 'Last Modified',   type: 'date',   columns: { rets_property: 'ModificationTimestamp' }, note: 'Drives incremental embedding updates (Week 6).' },

  // ---- Content / media ----
  remarks:    { label: 'Public Remarks', type: 'text', columns: { rets_property: 'L_Remarks' }, note: 'Has FULLTEXT index ft_remarks — use MATCH...AGAINST, not LIKE.' },
  photoCount: { label: 'Photo Count',    type: 'number', columns: { rets_property: 'PhotoCount' } },

  // ---- People ----
  listAgentFirst: { label: 'List Agent First Name', type: 'string', columns: { rets_property: 'LA1_UserFirstName', california_sold: 'ListAgentFirstName' } },
  listAgentLast:  { label: 'List Agent Last Name',  type: 'string', columns: { rets_property: 'LA1_UserLastName', california_sold: 'ListAgentLastName' } },
  listOffice:     { label: 'List Office',           type: 'string', columns: { rets_property: 'LO1_OrganizationName', california_sold: 'ListOfficeName' } },

  // ---- Open house (rets_openhouse) ----
  openHouseDate:  { label: 'Open House Date',  type: 'date', columns: { rets_openhouse: 'OpenHouseDate' } },
  openHouseStart: { label: 'Open House Start', type: 'string', columns: { rets_openhouse: 'OH_StartTime' } },
  openHouseEnd:   { label: 'Open House End',   type: 'string', columns: { rets_openhouse: 'OH_EndTime' } },
} satisfies Record<string, FieldDef>;

export type FieldName = keyof typeof FIELDS;

/**
 * Resolve a semantic field to its physical column on a given table.
 * Throws if the field does not exist on that table — fail fast at dev time
 * rather than producing a silently-wrong SQL string.
 */
export function col(field: FieldName, table: TableName): string {
  const cols: Partial<Record<TableName, string>> = FIELDS[field].columns;
  const c = cols[table];
  if (!c) {
    throw new Error(`Field '${field}' has no column on table '${table}'`);
  }
  return c;
}

/** Does this semantic field exist on the given table? */
export function hasField(field: FieldName, table: TableName): boolean {
  const cols: Partial<Record<TableName, string>> = FIELDS[field].columns;
  return Boolean(cols[table]);
}
