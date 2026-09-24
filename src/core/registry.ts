import type { DemoEntry } from './demo';
import { atmosphereEntry } from '../demos/atmosphere';
import { cloudsEntry } from '../demos/clouds';
import { oceanEntry } from '../demos/ocean';
import { planetEntry } from '../demos/planet';
import { watercolourEntry } from '../demos/watercolour';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [oceanEntry, cloudsEntry, watercolourEntry, atmosphereEntry, planetEntry];
