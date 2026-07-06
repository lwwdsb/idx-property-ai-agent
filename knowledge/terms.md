# Real-Estate Terms & Concepts

## Days on Market (DOM)
Days on Market is the number of days a listing has been active before it goes into
contract or closes. Lower DOM signals a hot market or an aggressively priced home;
high DOM can signal overpricing or weak demand. In this project the median DOM per
city is computed from california_sold over a trailing window.

## Price per Square Foot ($/sqft)
Price per square foot is the sale (or list) price divided by the living area in
square feet. It normalizes for home size, so it is the fairest way to compare homes
of different sizes. To estimate a fair value we multiply a city's median sold $/sqft
by the subject home's living area.

## List-to-Sold Ratio (sold-to-list)
The list-to-sold ratio is the close price divided by the original list price,
expressed as a percent. 100% means it sold at asking; above 100% means buyers bid
over asking (a seller's market); below 100% means homes sold under asking. We report
the median ratio per city over the trailing window.

## Comparable Sales (comps)
Comparables, or comps, are recently sold homes similar to a subject property, used to
estimate its fair value. Good comps share the same city, a similar living area
(this project uses ±20%), and a recent close date (trailing 6 months). Too few comps
(fewer than 3) means any price judgment is unreliable and should be withheld.

## Listing Status: Active, Pending, Sold
Active means the home is currently for sale (the rets_property table holds active
listings). Pending means an offer was accepted but the sale has not closed. Sold (or
closed) means the transaction completed (the california_sold table holds closed sales).

## HOA Fee
A Homeowners Association (HOA) fee is a recurring charge for shared community
amenities and maintenance (common in condos and planned developments). It is separate
from the mortgage and property tax and affects the true monthly cost of ownership.

## Living Area
Living area is the finished, heated interior square footage of a home. It excludes
garages, unfinished basements, and patios. It drives $/sqft and comp matching.

## Semantic vs Keyword Search
Semantic (vector) search matches meaning — "cozy craftsman with character" finds
homes described similarly even without those exact words. Keyword (BM25) search
matches exact terms — "solar panels", "ADU". This project fuses both with
Reciprocal Rank Fusion so each covers the other's blind spots.
