import { parseQuery } from './src/search/parseQuery.js';
import { isKnownCity } from './src/search/cityDictionary.js';
import { freshProfile, learnFromFilter, preferredFilter } from './src/memory/profile.js';
let p = freshProfile('u');
for (let i=0;i<3;i++){ const f=(await parseQuery('find 3-bed homes in Irvine under 2M', {isKnownCity})).filter; learnFromFilter(p,f); }
console.log('learned prefs:', JSON.stringify(p.prefs));
console.log('as soft defaults:', JSON.stringify(preferredFilter(p)));
