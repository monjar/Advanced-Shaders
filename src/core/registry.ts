import type { DemoEntry } from './demo';
import { atmosphereEntry } from '../demos/atmosphere';
import { blackholeEntry } from '../demos/blackhole';
import { cloudsEntry } from '../demos/clouds';
import { magicEntry } from '../demos/magic';
import { oceanEntry } from '../demos/ocean';
import { planetEntry } from '../demos/planet';
import { portalsEntry } from '../demos/portals';
import { sdfEntry } from '../demos/sdf';
import { watercolourEntry } from '../demos/watercolour';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [
  oceanEntry, cloudsEntry, watercolourEntry, portalsEntry, sdfEntry, blackholeEntry, magicEntry, atmosphereEntry, planetEntry,
];
