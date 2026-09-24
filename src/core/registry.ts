import type { DemoEntry } from './demo';
import { cloudsEntry } from '../demos/clouds';
import { oceanEntry } from '../demos/ocean';
import { watercolourEntry } from '../demos/watercolour';
import { blackholeEntry } from '../demos/blackhole';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [oceanEntry, cloudsEntry, watercolourEntry, blackholeEntry];
