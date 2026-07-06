# MLS Field Glossary

Generated from `schema/columns.ts` (single source of truth). Each semantic field
maps to physical column names per table.

## Listing / System ID (listingId)
- Type: string
- Columns: rets_property.L_ListingID, rets_openhouse.L_ListingID

## Listing Key (sold) (listingKey)
- Type: number
- Columns: california_sold.ListingKey
- Note: Join: CAST(rets_property.L_ListingID AS UNSIGNED) = california_sold.ListingKey

## MLS # (mlsNumber)
- Type: string
- Columns: rets_property.L_DisplayId, rets_openhouse.L_DisplayId

## Address (address)
- Type: string
- Columns: rets_property.L_Address, california_sold.UnparsedAddress

## City (city)
- Type: string
- Columns: rets_property.L_City, california_sold.City

## State (state)
- Type: string
- Columns: rets_property.L_State, california_sold.StateOrProvince

## ZIP / Postal Code (zip)
- Type: string
- Columns: rets_property.L_Zip, california_sold.PostalCode

## Subdivision (subdivision)
- Type: string
- Columns: rets_property.SubdivisionName, california_sold.SubdivisionName

## Latitude (latitude)
- Type: geo
- Columns: rets_property.LMD_MP_Latitude, california_sold.Latitude

## Longitude (longitude)
- Type: geo
- Columns: rets_property.LMD_MP_Longitude, california_sold.Longitude

## Property Type (propertyType)
- Type: string
- Columns: rets_property.L_Type_, california_sold.PropertyType

## Property Sub-Type (propertySubType)
- Type: string
- Columns: california_sold.PropertySubType

## Bedrooms (beds)
- Type: number
- Columns: rets_property.L_Keyword2, california_sold.BedroomsTotal

## Bathrooms (baths)
- Type: number
- Columns: rets_property.LM_Dec_3, california_sold.BathroomsTotalInteger

## Living Area (sqft) (livingArea)
- Type: number
- Columns: rets_property.LM_Int2_3, california_sold.LivingArea

## Year Built (yearBuilt)
- Type: number
- Columns: rets_property.YearBuilt, california_sold.YearBuilt

## Private Pool (pool)
- Type: flag
- Columns: rets_property.PoolPrivateYN, california_sold.PoolPrivateYN
- Note: Stored as varchar Y/N-ish; normalize before comparing.

## View (view)
- Type: flag
- Columns: rets_property.ViewYN, california_sold.ViewYN

## Fireplace (fireplace)
- Type: flag
- Columns: rets_property.FireplaceYN, california_sold.FireplaceYN

## HOA Fee (hoaFee)
- Type: number
- Columns: rets_property.AssociationFee, california_sold.AssociationFee

## List Price (price)
- Type: number
- Columns: rets_property.L_SystemPrice, california_sold.ListPrice

## Original List Price (originalListPrice)
- Type: number
- Columns: california_sold.OriginalListPrice

## Sold / Close Price (closePrice)
- Type: number
- Columns: california_sold.ClosePrice

## Listing Status (status)
- Type: string
- Columns: rets_property.L_Status
- Note: rets_property is all Active in current dump.

## Days on Market (daysOnMarket)
- Type: number
- Columns: rets_property.DaysOnMarket, california_sold.DaysOnMarket

## Listing Date (listingDate)
- Type: date
- Columns: rets_property.ListingContractDate, california_sold.ListingContractDate

## Close Date (closeDate)
- Type: date
- Columns: california_sold.CloseDate
- Note: Stored as VARCHAR with dirty values (e.g. year 2072) — clean/parse before use (Week 5).

## Last Modified (modifiedAt)
- Type: date
- Columns: rets_property.ModificationTimestamp
- Note: Drives incremental embedding updates (Week 6).

## Public Remarks (remarks)
- Type: text
- Columns: rets_property.L_Remarks
- Note: Has FULLTEXT index ft_remarks — use MATCH...AGAINST, not LIKE.

## Photo Count (photoCount)
- Type: number
- Columns: rets_property.PhotoCount

## List Agent First Name (listAgentFirst)
- Type: string
- Columns: rets_property.LA1_UserFirstName, california_sold.ListAgentFirstName

## List Agent Last Name (listAgentLast)
- Type: string
- Columns: rets_property.LA1_UserLastName, california_sold.ListAgentLastName

## List Office (listOffice)
- Type: string
- Columns: rets_property.LO1_OrganizationName, california_sold.ListOfficeName

## Open House Date (openHouseDate)
- Type: date
- Columns: rets_openhouse.OpenHouseDate

## Open House Start (openHouseStart)
- Type: string
- Columns: rets_openhouse.OH_StartTime

## Open House End (openHouseEnd)
- Type: string
- Columns: rets_openhouse.OH_EndTime
